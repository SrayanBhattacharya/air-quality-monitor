# ── inference.py — Model inference pipeline ────────────────────────────────────

import numpy as np
import pandas as pd
import torch
from datetime import timedelta
from fastapi import HTTPException

from config import (
    CSV_PATH, INPUT_WINDOW, OUTPUT_WINDOW,
    ROLL_WINDOW, NUMERIC_COLS
)
from preprocessing import preprocess, inverse_pm


def run_inference(model, scaler, device):
    """
    Read CSV, preprocess, run model, return inverse-transformed predictions.

    Returns:
        preds_inv  : np.ndarray shape (24, 2)  — absolute µg/m³ forecasts
        last_row   : pd.Series                 — latest raw sensor row
        timestamps : list[str]                 — forecast timestamps (HH:MM)
    """
    df_raw = pd.read_csv(CSV_PATH)

    # Validate enough rows exist
    required = ROLL_WINDOW + INPUT_WINDOW
    if len(df_raw) < required:
        raise HTTPException(
            status_code=400,
            detail=f"Not enough data. Need at least {required} rows, got {len(df_raw)}."
        )

    df = preprocess(df_raw)

    # Scale using training scaler
    df[NUMERIC_COLS] = scaler.transform(df[NUMERIC_COLS])

    # Take last INPUT_WINDOW rows as input sequence
    sequence = df[NUMERIC_COLS].values[-INPUT_WINDOW:]            # (48, 10)
    X = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0).to(device)  # (1, 48, 10)

    # Forward pass
    model.eval()
    with torch.no_grad():
        delta_pred = model(X)                                     # (1, 24, 2)

    # Convert delta predictions → absolute (still scaled)
    last_vals    = X[:, -1, :2]                                   # (1, 2)
    preds_scaled = (delta_pred + last_vals.unsqueeze(1)).squeeze(0).cpu().numpy()  # (24, 2)

    # Inverse transform to µg/m³
    pm25_inv  = inverse_pm(preds_scaled[:, 0], pm_index=0, scaler=scaler)
    pm10_inv  = inverse_pm(preds_scaled[:, 1], pm_index=1, scaler=scaler)
    preds_inv = np.stack([pm25_inv, pm10_inv], axis=1)            # (24, 2)

    # Generate forecast timestamps (15-min steps from last reading)
    last_row = df_raw.sort_values("datetime").iloc[-1]
    last_dt  = pd.to_datetime(last_row["datetime"]).tz_localize(None)
    timestamps = [
        (last_dt + timedelta(minutes=15 * (i + 1))).strftime("%H:%M")
        for i in range(OUTPUT_WINDOW)
    ]

    return preds_inv, last_row, timestamps