# AirWatch 🌫️

A real-time air quality monitoring and forecasting system for Kolkata, West Bengal. Live sensor data is collected via a Raspberry Pi, stored in Firebase, and fed into a Liquid Neural Network (LNN) model to predict PM2.5 and PM10 levels for the next 6 hours — displayed on a sleek dark dashboard with WHO threshold alerts.

---

## Features

- **Live sensor data** — PM2.5, PM10, Temperature, and Relative Humidity collected from a Raspberry Pi
- **6-hour forecast** — LNN model predicts the next 24 steps (15-minute intervals) of PM2.5 and PM10
- **WHO threshold alerts** — real-time warnings when pollutant levels are projected to exceed safe limits
- **2-hour trend** — rising/falling indicators for PM2.5 and PM10
- **Actual vs Predicted charts** — 12-hour comparison of model accuracy
- **Peak forecast** — shows the highest projected pollutant level and when it will occur
- **Live sync** — dashboard shows last sync time and online/offline status

---

## Architecture

```
Raspberry Pi (Sensors)
        │
        ▼
  Google Firebase
  (Realtime Database)
        │
        ▼
  FastAPI Backend
  (Fetches data, runs LNN model)
        │
        ▼
  React Frontend
  (Dashboard, charts, alerts)
```

---

## Tech Stack

**Hardware**

- Raspberry Pi
- PM2.5/PM10 sensor (SDS011)
- Temperature & Humidity sensor (BME280)

**Backend**

- Python
- FastAPI
- Liquid Neural Network (LNN) model
- Firebase Admin SDK
- NumPy / scikit-learn

**Frontend**

- React
- JavaScript
- Chart.js / Recharts (forecasting charts)
- Tailwind CSS

**Cloud**

- Google Firebase Realtime Database

---

## Project Structure

```
air-quality-monitor/
├── backend/
│   ├── app.py              # FastAPI server
│   ├── model.py            # LNN model definition and inference
│   ├── firebase_client.py  # Firebase data fetching
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── components/     # Dashboard components
│   │   └── App.jsx
│   └── package.json
└── .gitignore
```

---
