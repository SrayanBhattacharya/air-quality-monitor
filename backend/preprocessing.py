# ── preprocessing.py — Data preprocessing (must match training exactly) ────────

import numpy as np
import pandas as pd
from config import ROLL_WINDOW, NUMERIC_COLS


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Parse and sort datetime
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)

    # Coerce numeric
    for col in ["pm25", "pm10", "relativehumidity", "temperature"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Interpolate
    df = df.set_index("datetime")
    df[["pm25", "pm10", "relativehumidity", "temperature"]] = \
        df[["pm25", "pm10", "relativehumidity", "temperature"]].interpolate(method="time")
    df = df.reset_index()
    df = df.dropna().reset_index(drop=True)

    # Cyclical time encoding
    df["hour"]     = df["datetime"].dt.hour
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    df["dayofweek"] = df["datetime"].dt.dayofweek
    df["dow_sin"]   = np.sin(2 * np.pi * df["dayofweek"] / 7)
    df["dow_cos"]   = np.cos(2 * np.pi * df["dayofweek"] / 7)

    df = df.drop(columns=["hour", "dayofweek"])

    # Rolling means
    df["pm25_roll_mean_24"] = df["pm25"].rolling(window=ROLL_WINDOW, min_periods=ROLL_WINDOW).mean()
    df["pm10_roll_mean_24"] = df["pm10"].rolling(window=ROLL_WINDOW, min_periods=ROLL_WINDOW).mean()
    df = df.iloc[ROLL_WINDOW:].reset_index(drop=True)

    return df


def inverse_pm(values: np.ndarray, pm_index: int, scaler) -> np.ndarray:
    """Inverse transform a 1D array of scaled PM values back to µg/m³."""
    dummy = np.zeros((len(values), len(NUMERIC_COLS)))
    dummy[:, pm_index] = values
    return scaler.inverse_transform(dummy)[:, pm_index]