import { useCallback, useEffect, useState } from "react";
import {
  CartesianGrid,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

// ─── CONFIG ───────────────────────────────────────────────────────────────────
const API_URL =
  typeof process !== "undefined" && process.env?.REACT_APP_API_URL
    ? process.env.REACT_APP_API_URL
    : "http://localhost:8080";

const WHO_PM25 = 15;
const WHO_PM10 = 45;
const REFRESH_MS = 15 * 60 * 1000;

// ─── AQI HELPERS ──────────────────────────────────────────────────────────────
const AQI_BANDS = [
  { max: 12, label: "Good", color: "#4ade80", bg: "rgba(74,222,128,0.12)" },
  {
    max: 35.4,
    label: "Moderate",
    color: "#fbbf24",
    bg: "rgba(251,191,36,0.12)",
  },
  {
    max: 55.4,
    label: "Unhealthy for Sensitive",
    color: "#f97316",
    bg: "rgba(249,115,22,0.12)",
  },
  {
    max: 150.4,
    label: "Unhealthy",
    color: "#ef4444",
    bg: "rgba(239,68,68,0.12)",
  },
  {
    max: Infinity,
    label: "Very Unhealthy",
    color: "#a855f7",
    bg: "rgba(168,85,247,0.12)",
  },
];
const getAQI = (v) => AQI_BANDS.find((b) => v <= b.max) || AQI_BANDS.at(-1);

// ─── MOCK DATA ────────────────────────────────────────────────────────────────
const buildMock = () => {
  const now = new Date();
  const fmt = (d) => d.toTimeString().slice(0, 5);

  const history = Array.from({ length: 48 }, (_, i) => {
    const t = new Date(now - (47 - i) * 15 * 60 * 1000);
    return {
      ts: fmt(t),
      pm25: +(30 + Math.sin(i / 6) * 10 + Math.random() * 5).toFixed(1),
      pm10: +(60 + Math.sin(i / 5) * 15 + Math.random() * 8).toFixed(1),
      pm25pred: +(
        30 +
        Math.sin(i / 6) * 10 +
        (Math.random() - 0.5) * 8
      ).toFixed(1),
      pm10pred: +(
        60 +
        Math.sin(i / 5) * 15 +
        (Math.random() - 0.5) * 10
      ).toFixed(1),
    };
  });

  const forecast = Array.from({ length: 24 }, (_, i) => {
    const t = new Date(now + (i + 1) * 15 * 60 * 1000);
    return {
      ts: fmt(t),
      pm25: +(35 + Math.sin((i + 48) / 6) * 12 + Math.random() * 6).toFixed(1),
      pm10: +(65 + Math.sin((i + 48) / 5) * 18 + Math.random() * 9).toFixed(1),
    };
  });

  const last = history.at(-1);
  const peakPM25 = forecast.reduce((a, b) => (b.pm25 > a.pm25 ? b : a));
  const peakPM10 = forecast.reduce((a, b) => (b.pm10 > a.pm10 ? b : a));
  const peakPM25Idx = forecast.indexOf(peakPM25);
  const peakPM10Idx = forecast.indexOf(peakPM10);

  return {
    current: {
      datetime: now.toISOString(),
      pm25: last.pm25,
      pm10: last.pm10,
      temperature: 28.4,
      humidity: 65.2,
    },
    forecast: {
      timestamps: forecast.map((f) => f.ts),
      pm25: forecast.map((f) => f.pm25),
      pm10: forecast.map((f) => f.pm10),
    },
    history: {
      timestamps: history.map((h) => h.ts),
      pm25: history.map((h) => h.pm25),
      pm10: history.map((h) => h.pm10),
      pm25pred: history.map((h) => h.pm25pred),
      pm10pred: history.map((h) => h.pm10pred),
    },
    metrics: {
      pm25: { mae: 3.21, rmse: 4.85, r2: 0.91 },
      pm10: { mae: 5.1, rmse: 6.72, r2: 0.88 },
    },
    peak: {
      pm25: {
        value: peakPM25.pm25,
        timestamp: peakPM25.ts,
        minutes_from_now: (peakPM25Idx + 1) * 15,
        category: getAQI(peakPM25.pm25).label,
      },
      pm10: {
        value: peakPM10.pm10,
        timestamp: peakPM10.ts,
        minutes_from_now: (peakPM10Idx + 1) * 15,
        category: getAQI(peakPM10.pm10).label,
      },
    },
  };
};

// ─── FETCH (falls back to mock) ───────────────────────────────────────────────
const fetchAll = async () => {
  try {
    const [cur, fcast, hist, metr, peak] = await Promise.all([
      fetch(`${API_URL}/current`).then((r) => r.json()),
      fetch(`${API_URL}/forecast`).then((r) => r.json()),
      fetch(`${API_URL}/history`).then((r) => r.json()),
      fetch(`${API_URL}/metrics`).then((r) => r.json()),
      fetch(`${API_URL}/peak`).then((r) => r.json()),
    ]);
    return {
      current: cur,
      forecast: fcast,
      history: hist,
      metrics: metr,
      peak,
    };
  } catch {
    return buildMock();
  }
};

// ─── CUSTOM TOOLTIP ───────────────────────────────────────────────────────────
const DarkTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null;
  return (
    <div
      style={{
        background: "#0d1117",
        border: "1px solid #30363d",
        borderRadius: 6,
        padding: "8px 12px",
        fontFamily: "'IBM Plex Mono', monospace",
        fontSize: 11,
        color: "#c9d1d9",
      }}
    >
      <div style={{ color: "#8b949e", marginBottom: 4 }}>{label}</div>
      {payload.map((p) => (
        <div key={p.dataKey} style={{ color: p.color }}>
          {p.name}: {p.value} µg/m³
        </div>
      ))}
    </div>
  );
};

// ─── SPARKLINE ────────────────────────────────────────────────────────────────
const Sparkline = ({ data, color }) => (
  <ResponsiveContainer width="100%" height={36}>
    <LineChart data={data}>
      <Line
        type="monotone"
        dataKey="v"
        stroke={color}
        dot={false}
        strokeWidth={1.5}
      />
    </LineChart>
  </ResponsiveContainer>
);

// ─── TREND ARROW ─────────────────────────────────────────────────────────────
const getTrend = (hist, key) => {
  const recent = hist.slice(-4).map((h) => h[key]);
  const older = hist.slice(-8, -4).map((h) => h[key]);
  const avgR = recent.reduce((a, b) => a + b, 0) / recent.length;
  const avgO = older.reduce((a, b) => a + b, 0) / older.length;
  const diff = avgR - avgO;
  if (diff > 1.5) return { arrow: "↑", label: "Rising", color: "#ef4444" };
  if (diff < -1.5) return { arrow: "↓", label: "Falling", color: "#4ade80" };
  return { arrow: "→", label: "Stable", color: "#fbbf24" };
};

const fmtMins = (m) => {
  const h = Math.floor(m / 60);
  const min = m % 60;
  if (h === 0) return `${min}m`;
  return min === 0 ? `${h}h` : `${h}h ${min}m`;
};

// ─── MAIN COMPONENT ───────────────────────────────────────────────────────────
export default function AirQualityDashboard() {
  const [data, setData] = useState(null);
  const [lastFetch, setLastFetch] = useState(null);
  const [loading, setLoading] = useState(true);

  const load = useCallback(async () => {
    const d = await fetchAll();
    setData(d);
    setLastFetch(new Date());
    setLoading(false);
  }, []);

  useEffect(() => {
    load();
    const iv = setInterval(load, REFRESH_MS);
    return () => clearInterval(iv);
  }, [load]);

  if (loading || !data) {
    return (
      <div style={styles.loadWrap}>
        <div style={styles.loadText}>INITIALIZING SENSOR ARRAY…</div>
      </div>
    );
  }

  const { current, forecast, history, metrics, peak } = data;

  // Build chart arrays
  const forecastChartData = forecast.timestamps.map((ts, i) => ({
    ts,
    pm25: forecast.pm25[i],
    pm10: forecast.pm10[i],
  }));
  const histChartData = history.timestamps.map((ts, i) => ({
    ts,
    pm25: history.pm25[i],
    pm10: history.pm10[i],
    pm25p: history.pm25pred?.[i],
    pm10p: history.pm10pred?.[i],
  }));

  // Sparkline: last 8 readings (2h)
  const spPM25 = histChartData.slice(-8).map((h) => ({ v: h.pm25 }));
  const spPM10 = histChartData.slice(-8).map((h) => ({ v: h.pm10 }));

  const trendPM25 = getTrend(histChartData, "pm25");
  const trendPM10 = getTrend(histChartData, "pm10");

  // R² per forecast step mock
  const r2Steps = Array.from({ length: 24 }, (_, i) => ({
    step: i + 1,
    pm25: +(metrics.pm25.r2 - i * 0.008 + (Math.random() - 0.5) * 0.02).toFixed(
      3,
    ),
    pm10: +(metrics.pm10.r2 - i * 0.009 + (Math.random() - 0.5) * 0.02).toFixed(
      3,
    ),
  }));

  // Sensor alive?
  const isLive = current.status === "live";

  // Alert?
  const pm25Alert = forecast.pm25.some((v) => v > WHO_PM25);
  const pm10Alert = forecast.pm10.some((v) => v > WHO_PM10);
  const alertTs = pm25Alert
    ? forecast.timestamps[forecast.pm25.findIndex((v) => v > WHO_PM25)]
    : null;
  const alertTs10 = pm10Alert
    ? forecast.timestamps[forecast.pm10.findIndex((v) => v > WHO_PM10)]
    : null;

  const pm25AQI = getAQI(current.pm25);
  const pm10AQI = getAQI(current.pm10);

  return (
    <div style={styles.root}>
      {/* Google Font */}
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;500;600&family=Space+Grotesk:wght@300;400;500;600;700&display=swap');
        * { box-sizing: border-box; margin: 0; padding: 0; }
        ::-webkit-scrollbar { width: 4px; } ::-webkit-scrollbar-track { background: #0d1117; }
        ::-webkit-scrollbar-thumb { background: #30363d; border-radius: 2px; }
        @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.4} }
        @keyframes fadeIn { from{opacity:0;transform:translateY(8px)} to{opacity:1;transform:translateY(0)} }
        .card { animation: fadeIn 0.4s ease both; }
        .live-dot { animation: pulse 2s infinite; }
      `}</style>

      {/* ── HEADER ── */}
      <header style={styles.header}>
        <div style={styles.headerLeft}>
          <div style={styles.logo}>
            <span style={styles.logoIcon}>◉</span>
            <span style={styles.logoText}>AIRWATCH</span>
          </div>
          <div style={styles.location}>📍 Kolkata, West Bengal</div>
        </div>
        <div style={styles.headerRight}>
          <div style={styles.statusBadge(isLive)}>
            <span
              className="live-dot"
              style={{
                width: 7,
                height: 7,
                borderRadius: "50%",
                background: isLive ? "#4ade80" : "#ef4444",
                display: "inline-block",
              }}
            />
            {isLive ? "LIVE" : "OFFLINE"}
          </div>
          <div style={styles.lastUpdated}>
            LAST SYNC {lastFetch?.toTimeString().slice(0, 8)}
          </div>
        </div>
      </header>

      <div style={styles.main}>
        {/* ── ALERT BANNER ── */}
        {(pm25Alert || pm10Alert) && (
          <div style={styles.alertBanner} className="card">
            <span style={{ fontSize: 16 }}>⚠</span>
            <span>
              {pm25Alert &&
                `PM2.5 projected to exceed WHO limit (${WHO_PM25} µg/m³) at ${alertTs}`}
              {pm25Alert && pm10Alert && "  ·  "}
              {pm10Alert &&
                `PM10 projected to exceed WHO limit (${WHO_PM10} µg/m³) at ${alertTs10}`}
            </span>
          </div>
        )}

        {/* ── CURRENT READINGS ── */}
        <div style={styles.row4}>
          {[
            {
              label: "PM2.5",
              value: current.pm25,
              unit: "µg/m³",
              aqi: pm25AQI,
            },
            { label: "PM10", value: current.pm10, unit: "µg/m³", aqi: pm10AQI },
            {
              label: "TEMP",
              value: current.temperature,
              unit: "°C",
              aqi: null,
            },
            { label: "RH", value: current.humidity, unit: "%", aqi: null },
          ].map(({ label, value, unit, aqi }, i) => (
            <div
              key={label}
              className="card"
              style={{
                ...styles.card,
                animationDelay: `${i * 60}ms`,
                borderTop: aqi ? `2px solid ${aqi.color}` : "2px solid #30363d",
              }}
            >
              <div style={styles.cardLabel}>{label}</div>
              <div style={styles.cardValue}>{value}</div>
              <div style={styles.cardUnit}>{unit}</div>
              {aqi && (
                <div
                  style={{
                    ...styles.aqiBadge,
                    background: aqi.bg,
                    color: aqi.color,
                  }}
                >
                  {aqi.label}
                </div>
              )}
            </div>
          ))}
        </div>

        {/* ── PEAK + TREND ROW ── */}
        <div style={styles.row2}>
          {/* Peak Forecast */}
          <div className="card" style={styles.card}>
            <div style={styles.sectionTitle}>PEAK FORECAST · NEXT 6H</div>
            <div style={styles.peakGrid}>
              {["pm25", "pm10"].map((key) => {
                const p = peak[key];
                const aqi = getAQI(p.value);
                const safe =
                  key === "pm25" ? p.value <= WHO_PM25 : p.value <= WHO_PM10;
                return (
                  <div
                    key={key}
                    style={{
                      ...styles.peakItem,
                      background: safe
                        ? "rgba(74,222,128,0.05)"
                        : "rgba(239,68,68,0.06)",
                      borderLeft: `3px solid ${safe ? "#4ade80" : aqi.color}`,
                    }}
                  >
                    <div style={styles.peakLabel}>
                      {safe ? "✓" : "⚠"} {key.toUpperCase()}
                    </div>
                    <div style={{ ...styles.peakValue, color: aqi.color }}>
                      {p.value}{" "}
                      <span style={{ fontSize: 11, opacity: 0.7 }}>µg/m³</span>
                    </div>
                    <div style={styles.peakMeta}>
                      {safe
                        ? "Within safe limits"
                        : `At ${p.timestamp} · in ${fmtMins(p.minutes_from_now)}`}
                    </div>
                    {!safe && (
                      <div
                        style={{
                          ...styles.aqiBadge,
                          background: aqi.bg,
                          color: aqi.color,
                          marginTop: 6,
                        }}
                      >
                        {aqi.label}
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          </div>

          {/* Trend Indicators */}
          <div className="card" style={styles.card}>
            <div style={styles.sectionTitle}>2-HOUR TREND</div>
            {[
              {
                key: "pm25",
                label: "PM2.5",
                trend: trendPM25,
                sp: spPM25,
                color: "#38bdf8",
              },
              {
                key: "pm10",
                label: "PM10",
                trend: trendPM10,
                sp: spPM10,
                color: "#a78bfa",
              },
            ].map(({ label, trend, sp, color }) => (
              <div key={label} style={styles.trendRow}>
                <div style={styles.trendLabel}>{label}</div>
                <div style={{ flex: 1 }}>
                  <Sparkline data={sp} color={color} />
                </div>
                <div
                  style={{
                    color: trend.color,
                    fontFamily: "'IBM Plex Mono'",
                    fontSize: 22,
                    width: 32,
                    textAlign: "center",
                  }}
                >
                  {trend.arrow}
                </div>
                <div style={{ ...styles.trendBadge, color: trend.color }}>
                  {trend.label}
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* ── 6H FORECAST CHART ── */}
        <div className="card" style={styles.card}>
          <div style={styles.sectionTitle}>
            6-HOUR FORECAST · 15MIN INTERVALS
          </div>
          <div style={styles.legendRow}>
            <span style={styles.legend("#38bdf8")}>— PM2.5</span>
            <span style={styles.legend("#a78bfa")}>— PM10</span>
            <span style={styles.legend("#ef4444", true)}>
              - - WHO Threshold
            </span>
          </div>
          <ResponsiveContainer width="100%" height={220}>
            <LineChart
              data={forecastChartData}
              margin={{ top: 8, right: 20, bottom: 0, left: 0 }}
            >
              <CartesianGrid stroke="#161b22" strokeDasharray="3 3" />
              <XAxis
                dataKey="ts"
                tick={{
                  fill: "#8b949e",
                  fontSize: 9,
                  fontFamily: "IBM Plex Mono",
                }}
                interval={3}
              />
              <YAxis
                tick={{
                  fill: "#8b949e",
                  fontSize: 9,
                  fontFamily: "IBM Plex Mono",
                }}
              />
              <Tooltip content={<DarkTooltip />} />
              <ReferenceLine
                y={WHO_PM25}
                stroke="#ef4444"
                strokeDasharray="4 2"
                strokeOpacity={0.6}
                label={{ value: "WHO PM2.5", fill: "#ef4444", fontSize: 9 }}
              />
              <ReferenceLine
                y={WHO_PM10}
                stroke="#f97316"
                strokeDasharray="4 2"
                strokeOpacity={0.6}
                label={{ value: "WHO PM10", fill: "#f97316", fontSize: 9 }}
              />
              <Line
                type="monotone"
                dataKey="pm25"
                name="PM2.5"
                stroke="#38bdf8"
                dot={false}
                strokeWidth={2}
              />
              <Line
                type="monotone"
                dataKey="pm10"
                name="PM10"
                stroke="#a78bfa"
                dot={false}
                strokeWidth={2}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>

        {/* ── HISTORY CHARTS ── */}
        <div style={styles.row2}>
          {[
            {
              key: "pm25",
              keyP: "pm25p",
              label: "PM2.5",
              c1: "#38bdf8",
              c2: "#0ea5e9",
            },
            {
              key: "pm10",
              keyP: "pm10p",
              label: "PM10",
              c1: "#a78bfa",
              c2: "#7c3aed",
            },
          ].map(({ key, keyP, label, c1, c2 }) => (
            <div className="card" key={key} style={styles.card}>
              <div style={styles.sectionTitle}>
                ACTUAL vs PREDICTED · {label} · 12H
              </div>
              <div style={styles.legendRow}>
                <span style={styles.legend(c1)}>— Actual</span>
                <span style={styles.legend(c2, false, true)}>·· Predicted</span>
              </div>
              <ResponsiveContainer width="100%" height={160}>
                <LineChart
                  data={histChartData}
                  margin={{ top: 4, right: 16, bottom: 0, left: 0 }}
                >
                  <CartesianGrid stroke="#161b22" strokeDasharray="3 3" />
                  <XAxis
                    dataKey="ts"
                    tick={{
                      fill: "#8b949e",
                      fontSize: 8,
                      fontFamily: "IBM Plex Mono",
                    }}
                    interval={7}
                  />
                  <YAxis
                    tick={{
                      fill: "#8b949e",
                      fontSize: 8,
                      fontFamily: "IBM Plex Mono",
                    }}
                  />
                  <Tooltip content={<DarkTooltip />} />
                  <Line
                    type="monotone"
                    dataKey={key}
                    name="Actual"
                    stroke={c1}
                    dot={false}
                    strokeWidth={1.5}
                  />
                  <Line
                    type="monotone"
                    dataKey={keyP}
                    name="Predicted"
                    stroke={c2}
                    dot={false}
                    strokeWidth={1.5}
                    strokeDasharray="3 2"
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          ))}
        </div>

        <div style={styles.footer}>
          AIRWATCH v0.1 · Liquid Neural Network Forecast · Auto-refresh every
          15min · {new Date().getFullYear()}
        </div>
      </div>
    </div>
  );
}

// ─── STYLES ───────────────────────────────────────────────────────────────────
const styles = {
  root: {
    minHeight: "100vh",
    background: "#0d1117",
    color: "#c9d1d9",
    fontFamily: "'Space Grotesk', sans-serif",
  },
  loadWrap: {
    minHeight: "100vh",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    background: "#0d1117",
  },
  loadText: {
    fontFamily: "'IBM Plex Mono', monospace",
    color: "#38bdf8",
    fontSize: 14,
    letterSpacing: 4,
  },
  header: {
    display: "flex",
    alignItems: "center",
    justifyContent: "space-between",
    padding: "14px 28px",
    borderBottom: "1px solid #161b22",
    background: "#010409",
    position: "sticky",
    top: 0,
    zIndex: 100,
  },
  headerLeft: { display: "flex", alignItems: "center", gap: 20 },
  logo: { display: "flex", alignItems: "center", gap: 10 },
  logoIcon: { color: "#38bdf8", fontSize: 22 },
  logoText: {
    fontFamily: "'IBM Plex Mono', monospace",
    fontWeight: 600,
    fontSize: 15,
    letterSpacing: 6,
    color: "#e6edf3",
  },
  location: {
    color: "#8b949e",
    fontSize: 12,
    fontFamily: "'IBM Plex Mono', monospace",
  },
  headerRight: { display: "flex", alignItems: "center", gap: 16 },
  statusBadge: (live) => ({
    display: "flex",
    alignItems: "center",
    gap: 6,
    background: live ? "rgba(74,222,128,0.1)" : "rgba(239,68,68,0.1)",
    border: `1px solid ${live ? "#4ade80" : "#ef4444"}`,
    color: live ? "#4ade80" : "#ef4444",
    padding: "3px 10px",
    borderRadius: 4,
    fontFamily: "'IBM Plex Mono', monospace",
    fontSize: 11,
    fontWeight: 600,
    letterSpacing: 1,
  }),
  lastUpdated: {
    fontFamily: "'IBM Plex Mono', monospace",
    color: "#8b949e",
    fontSize: 10,
  },
  main: {
    padding: "20px 24px",
    maxWidth: 1400,
    margin: "0 auto",
    display: "flex",
    flexDirection: "column",
    gap: 16,
  },
  alertBanner: {
    display: "flex",
    alignItems: "center",
    gap: 10,
    background: "rgba(239,68,68,0.08)",
    border: "1px solid rgba(239,68,68,0.4)",
    borderLeft: "4px solid #ef4444",
    borderRadius: 6,
    padding: "12px 16px",
    fontFamily: "'IBM Plex Mono', monospace",
    fontSize: 12,
    color: "#ef4444",
  },
  row4: { display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 16 },
  row2: { display: "grid", gridTemplateColumns: "repeat(2, 1fr)", gap: 16 },
  card: {
    background: "#0d1117",
    border: "1px solid #21262d",
    borderRadius: 8,
    padding: "18px 20px",
  },
  cardLabel: {
    fontFamily: "'IBM Plex Mono', monospace",
    fontSize: 10,
    color: "#8b949e",
    letterSpacing: 2,
    marginBottom: 8,
  },
  cardValue: {
    fontFamily: "'IBM Plex Mono', monospace",
    fontSize: 32,
    fontWeight: 600,
    color: "#e6edf3",
    lineHeight: 1,
  },
  cardUnit: {
    fontFamily: "'IBM Plex Mono', monospace",
    fontSize: 10,
    color: "#8b949e",
    marginTop: 4,
    marginBottom: 8,
  },
  aqiBadge: {
    display: "inline-block",
    padding: "2px 8px",
    borderRadius: 3,
    fontSize: 10,
    fontWeight: 600,
    fontFamily: "'IBM Plex Mono', monospace",
    letterSpacing: 0.5,
  },
  sectionTitle: {
    fontFamily: "'IBM Plex Mono', monospace",
    fontSize: 10,
    color: "#8b949e",
    letterSpacing: 2,
    marginBottom: 14,
  },
  legendRow: { display: "flex", gap: 16, marginBottom: 10 },
  legend: (color, dashed, dotted) => ({
    fontFamily: "'IBM Plex Mono', monospace",
    fontSize: 10,
    color,
    borderBottom: dashed
      ? `1px dashed ${color}`
      : dotted
        ? `1px dotted ${color}`
        : `1px solid ${color}`,
    paddingBottom: 1,
  }),
  peakGrid: { display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 },
  peakItem: { borderRadius: 6, padding: "12px 14px" },
  peakLabel: {
    fontFamily: "'IBM Plex Mono', monospace",
    fontSize: 10,
    color: "#8b949e",
    letterSpacing: 1,
  },
  peakValue: {
    fontFamily: "'IBM Plex Mono', monospace",
    fontSize: 24,
    fontWeight: 600,
    marginTop: 4,
  },
  peakMeta: {
    fontFamily: "'IBM Plex Mono', monospace",
    fontSize: 10,
    color: "#8b949e",
    marginTop: 4,
  },
  trendRow: { display: "flex", alignItems: "center", gap: 12, marginBottom: 8 },
  trendLabel: {
    fontFamily: "'IBM Plex Mono', monospace",
    fontSize: 10,
    color: "#8b949e",
    width: 40,
  },
  trendBadge: {
    fontFamily: "'IBM Plex Mono', monospace",
    fontSize: 10,
    width: 50,
  },
  metricRow: {
    display: "flex",
    justifyContent: "space-between",
    marginBottom: 8,
    gap: 16,
  },
  metricKey: {
    fontFamily: "'IBM Plex Mono', monospace",
    fontSize: 11,
    color: "#8b949e",
  },
  metricVal: {
    fontFamily: "'IBM Plex Mono', monospace",
    fontSize: 13,
    fontWeight: 600,
  },
  footer: {
    fontFamily: "'IBM Plex Mono', monospace",
    fontSize: 9,
    color: "#484f58",
    textAlign: "center",
    letterSpacing: 1,
    paddingTop: 8,
    paddingBottom: 20,
  },
};
