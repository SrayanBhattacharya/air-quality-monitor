# ── config.py — All constants (must match training exactly) ───────────────────

INPUT_WINDOW  = 48       # 12 hours of 15-min readings
OUTPUT_WINDOW = 24       # 6 hours ahead
ROLL_WINDOW   = 24       # rolling mean window
NUM_TARGETS   = 2        # PM2.5 and PM10
HIDDEN_SIZE   = 128
INPUT_SIZE    = 10

NUMERIC_COLS = [
    "pm25", "pm10", "temperature", "relativehumidity",
    "hour_sin", "hour_cos",
    "dow_sin", "dow_cos",
    "pm25_roll_mean_24", "pm10_roll_mean_24"
]

CSV_PATH    = "sensor_data.csv"
MODEL_PATH  = "models/LNN.pth"
SCALER_PATH = "models/scaler.pkl"

# WHO thresholds (µg/m³)
PM25_THRESHOLD = 15.0
PM10_THRESHOLD = 45.0

# ── Firebase ──────────────────────────────────────────────────────────────────
FIREBASE_CRED_PATH = "real-time-weather-b3d71-firebase-adminsdk-fbsvc-89ef89a785.json"
FIREBASE_DB_URL    = "https://real-time-weather-b3d71-default-rtdb.firebaseio.com/"
FIREBASE_NODE      = "env_data_new"
 
# Fetch a few extra rows beyond minimum (ROLL_WINDOW + INPUT_WINDOW = 72)
FIREBASE_FETCH_LIMIT = 80