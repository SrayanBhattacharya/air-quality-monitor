# ── main.py — FastAPI app, lifespan, and endpoints ────────────────────────────

import torch
import joblib
import numpy as np
import pandas as pd
import firebase_admin
from firebase_admin import credentials, db
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import (
    MODEL_PATH, SCALER_PATH,
    INPUT_SIZE, HIDDEN_SIZE, OUTPUT_WINDOW, NUM_TARGETS, INPUT_WINDOW,
    PM25_THRESHOLD, PM10_THRESHOLD,
    FIREBASE_CRED_PATH, FIREBASE_DB_URL, FIREBASE_NODE, FIREBASE_FETCH_LIMIT
)
from model import LiquidPMModel
from inference import run_inference


# ── Global State ──────────────────────────────────────────────────────────────
model  = None
scaler = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Helpers ───────────────────────────────────────────────────────────────────
def get_aqi_category(pm25_value: float) -> str:
    if pm25_value <= 12.0:
        return "Good"
    elif pm25_value <= 35.4:
        return "Moderate"
    elif pm25_value <= 55.4:
        return "Unhealthy for Sensitive Groups"
    elif pm25_value <= 150.4:
        return "Unhealthy"
    else:
        return "Very Unhealthy"


def fetch_latest_records(n: int) -> pd.DataFrame:
    """Fetch latest n records from Firebase."""
    ref  = db.reference(FIREBASE_NODE)
    data = ref.order_by_key().limit_to_last(n).get()
    if not data:
        return pd.DataFrame()
    df = pd.DataFrame(list(data.values()))
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)
    return df


# ── App Lifespan ──────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, scaler

    # Init Firebase
    print("Initializing Firebase...")
    cred = credentials.Certificate(FIREBASE_CRED_PATH)
    firebase_admin.initialize_app(cred, {"databaseURL": FIREBASE_DB_URL})
    print("Firebase connected!")

    # Load scaler
    print("Loading scaler...")
    scaler = joblib.load(SCALER_PATH)

    # Load model
    print("Loading model...")
    model = LiquidPMModel(
        input_size=INPUT_SIZE,
        hidden_size=HIDDEN_SIZE,
        output_steps=OUTPUT_WINDOW,
        num_targets=NUM_TARGETS
    ).to(device)
    state_dict = torch.load(MODEL_PATH, map_location=device)
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()
    print(f"Model loaded on {device}")

    yield  # app runs here

    print("Shutting down...")


# ── FastAPI App ────────────────────────────────────────────────────────────────
app = FastAPI(title="Air Quality Dashboard API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in production
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "device": str(device)}


@app.get("/current")
def get_current():
    """Latest sensor reading from Firebase."""
    df   = fetch_latest_records(1)
    if df.empty:
        return {"error": "No data available"}

    last = df.iloc[-1]

    # Sensor freshness check
    last_dt       = pd.to_datetime(last["datetime"]).tz_localize(None)
    now           = pd.Timestamp.utcnow().tz_localize(None)
    minutes_since = (now - last_dt).total_seconds() / 60
    status        = "live" if minutes_since <= 20 else "offline"

    return {
        "datetime":     last["datetime"].strftime("%Y-%m-%dT%H:%M:%S"),
        "pm25":         round(float(last["pm25"]), 2),
        "pm10":         round(float(last["pm10"]), 2),
        "temperature":  round(float(last["temperature"]), 2),
        "humidity":     round(float(last["relativehumidity"]), 2),
        "status":       status,
        "aqi_category": get_aqi_category(float(last["pm25"]))
    }


@app.get("/forecast")
def get_forecast():
    """24-step (6-hour) PM2.5 and PM10 forecast."""
    preds_inv, _, timestamps = run_inference(model, scaler, device)

    return {
        "timestamps": timestamps,
        "pm25": [round(float(v), 2) for v in preds_inv[:, 0]],
        "pm10": [round(float(v), 2) for v in preds_inv[:, 1]],
    }


@app.get("/peak")
def get_peak():
    """Peak predicted PM2.5 and PM10 values in the next 6 hours."""
    preds_inv, _, timestamps = run_inference(model, scaler, device)

    def peak_info(values, timestamps, threshold):
        peak_idx   = int(np.argmax(values))
        peak_val   = float(values[peak_idx])
        minutes    = (peak_idx + 1) * 15
        hours      = minutes // 60
        mins_rem   = minutes % 60
        time_label = f"{hours}h {mins_rem}min" if hours > 0 else f"{mins_rem}min"

        return {
            "value":             round(peak_val, 2),
            "timestamp":         timestamps[peak_idx],
            "minutes_from_now":  minutes,
            "time_label":        f"in ~{time_label}",
            "category":          get_aqi_category(peak_val),
            "exceeds_threshold": peak_val > threshold
        }

    return {
        "pm25": peak_info(preds_inv[:, 0], timestamps, PM25_THRESHOLD),
        "pm10": peak_info(preds_inv[:, 1], timestamps, PM10_THRESHOLD),
    }


@app.get("/history")
def get_history():
    """Last 48 readings (12 hours) of true sensor values from Firebase."""
    df = fetch_latest_records(INPUT_WINDOW)
    if df.empty:
        return {"timestamps": [], "pm25": [], "pm10": []}

    return {
        "timestamps": df["datetime"].dt.strftime("%H:%M").tolist(),
        "pm25":       [round(float(v), 2) for v in df["pm25"]],
        "pm10":       [round(float(v), 2) for v in df["pm10"]],
    }


@app.get("/metrics")
def get_metrics():
    """
    Model performance metrics from training.
    Replace these with your actual results from the notebook.
    """
    return {
        "pm25": {"mae": 3.21, "rmse": 4.85, "r2": 0.91},
        "pm10": {"mae": 5.10, "rmse": 6.72, "r2": 0.88},
    }


@app.get("/config")
def get_config():
    """Frontend configuration."""
    return {
        "location": "Kolkata Monitoring Station"
    }