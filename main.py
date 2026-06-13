from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import datetime
import math
import time

app = FastAPI(
    title="AGOS Mock Server — Demo Mode",
    description="Mock backend simulating flood prediction responses for demo/presentation purposes."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Helper: which scenario are we in right now?
# Full cycle = 270s (90s normal → 90s advisory → 90s warning → repeat)
# ---------------------------------------------------------------------------
def current_scenario() -> str:
    position = time.time() % 270
    if position < 90:
        return "normal"
    elif position < 180:
        return "advisory"
    else:
        return "warning"


# ---------------------------------------------------------------------------
# Helper: smooth 0.0–1.0 oscillation within the current 90s window
# ---------------------------------------------------------------------------
def cycle_factor() -> float:
    position = time.time() % 90
    return (math.sin((position / 90.0) * 2 * math.pi) + 1) / 2


def lerp(a: float, b: float, t: float) -> float:
    return round(a + (b - a) * t, 2)


# ---------------------------------------------------------------------------
# Helper: alert level string from probability
# ---------------------------------------------------------------------------
def alert_level(prob: float) -> str:
    if prob >= 0.60:
        return "WARNING"
    elif prob >= 0.35:
        return "ADVISORY"
    return "NORMAL"


# ---------------------------------------------------------------------------
# Helper: live metrics per scenario
# ---------------------------------------------------------------------------
def live_metrics():
    f = cycle_factor()
    s = current_scenario()

    if s == "warning":
        return {
            "rainfall_mm":  lerp(18.0, 30.0, f),
            "wind_signal":  2,
            "max_wind_kph": lerp(65.0, 85.0, f),
            "humidity":     lerp(92.0, 98.0, f),
        }
    elif s == "advisory":
        return {
            "rainfall_mm":  lerp(6.0, 14.0, f),
            "wind_signal":  1,
            "max_wind_kph": lerp(40.0, 62.0, f),
            "humidity":     lerp(82.0, 91.0, f),
        }
    else:  # normal
        return {
            "rainfall_mm":  lerp(0.0, 1.5, f),
            "wind_signal":  0,
            "max_wind_kph": lerp(8.0, 18.0, f),
            "humidity":     lerp(70.0, 82.0, f),
        }


def live_probability() -> float:
    f = cycle_factor()
    s = current_scenario()

    if s == "warning":
        return round(lerp(0.65, 0.92, f), 4)
    elif s == "advisory":
        return round(lerp(0.35, 0.55, f), 4)
    else:
        return round(lerp(0.05, 0.20, f), 4)


# ---------------------------------------------------------------------------
# Helper: generate mock hourly entries
# ---------------------------------------------------------------------------
def mock_hourly():
    f = cycle_factor()
    s = current_scenario()

    if s == "warning":
        base_precip = lerp(6.0, 12.0, f)
        base_wind   = lerp(65.0, 80.0, f)
        base_humid  = lerp(91.0, 97.0, f)
        base_temp   = lerp(22.0, 25.0, f)
        wmo         = 63
        condition   = "Moderate rain"
    elif s == "advisory":
        base_precip = lerp(2.0, 5.0, f)
        base_wind   = lerp(40.0, 60.0, f)
        base_humid  = lerp(82.0, 90.0, f)
        base_temp   = lerp(25.0, 27.5, f)
        wmo         = 51
        condition   = "Light drizzle"
    else:
        base_precip = lerp(0.0, 0.8, f)
        base_wind   = lerp(8.0, 18.0, f)
        base_humid  = lerp(70.0, 81.0, f)
        base_temp   = lerp(27.0, 29.5, f)
        wmo         = 2
        condition   = "Partly cloudy"

    now = datetime.datetime.now().replace(minute=0, second=0, microsecond=0)
    hourly = []
    for i in range(48):
        t = now + datetime.timedelta(hours=i)
        spike    = (base_precip * 2.5) if (6 <= i <= 18 and s == "warning") else base_precip
        wind_val = round(base_wind + (i % 4) * 1.2, 1)
        hourly.append({
            "time":           t.strftime("%Y-%m-%dT%H:%M"),
            "precipitation":  round(spike + (i % 3) * 0.1, 2),
            "humidity":       round(base_humid + (i % 5) * 0.4, 1),
            "wind_speed_kph": wind_val,
            "temperature_c":  round(base_temp - (i % 6) * 0.2, 1),
            "weathercode":    wmo,
            "condition":      condition,
            "wind_signal":    2 if wind_val >= 62 else (1 if wind_val >= 39 else 0),
        })
    return hourly


# ---------------------------------------------------------------------------
# Helper: generate mock daily entries
# ---------------------------------------------------------------------------
def mock_daily():
    f = cycle_factor()
    s = current_scenario()
    today = datetime.date.today()

    if s == "warning":
        daily_precip = [lerp(30.0, 45.0, f), lerp(40.0, 58.0, f), 28.0, 12.0, 5.5]
        daily_prob   = [int(lerp(80, 93, f)), int(lerp(88, 96, f)), 75, 60, 40]
        daily_wind   = [lerp(62.0, 78.0, f), lerp(68.0, 82.0, f), 55.0, 40.0, 28.0]
        daily_wmo    = [63, 65, 61, 51, 2]
        conditions   = ["Moderate rain", "Heavy rain", "Moderate rain", "Light drizzle", "Partly cloudy"]
    elif s == "advisory":
        daily_precip = [lerp(10.0, 20.0, f), lerp(8.0, 16.0, f), 12.0, 6.0, 2.0]
        daily_prob   = [int(lerp(45, 65, f)), int(lerp(40, 60, f)), 50, 35, 20]
        daily_wind   = [lerp(38.0, 58.0, f), lerp(35.0, 55.0, f), 42.0, 30.0, 20.0]
        daily_wmo    = [51, 53, 51, 2, 1]
        conditions   = ["Light drizzle", "Moderate drizzle", "Light drizzle", "Partly cloudy", "Mainly clear"]
    else:
        daily_precip = [lerp(0.5, 3.0, f), lerp(0.2, 2.0, f), 0.5, 2.5, 1.0]
        daily_prob   = [int(lerp(10, 25, f)), int(lerp(8, 18, f)), 10, 22, 12]
        daily_wind   = [lerp(8.0, 18.0, f), lerp(6.0, 15.0, f), 10.0, 14.0, 9.0]
        daily_wmo    = [2, 1, 0, 2, 1]
        conditions   = ["Partly cloudy", "Mainly clear", "Clear sky", "Partly cloudy", "Mainly clear"]

    daily = []
    for i in range(5):
        d = today + datetime.timedelta(days=i)
        w = daily_wind[i]
        daily.append({
            "date":                 d.strftime("%Y-%m-%d"),
            "precipitation_sum_mm": round(daily_precip[i], 1),
            "rain_probability_pct": daily_prob[i],
            "temperature_max_c":    round(lerp(26.0, 29.5, f) - i * 0.3, 1),
            "wind_speed_max_kph":   round(w, 1),
            "wind_signal":          2 if w >= 62 else (1 if w >= 39 else 0),
            "weathercode":          daily_wmo[i],
            "condition":            conditions[i],
        })
    return daily


# ---------------------------------------------------------------------------
# GET /api/predict-flood
# ---------------------------------------------------------------------------
@app.get("/api/predict-flood")
def predict_flood():
    metrics = live_metrics()
    prob    = live_probability()
    s       = current_scenario()
    return {
        "status": "success",
        "alert_level": alert_level(prob),
        "probability": prob,
        "lead_time_estimate": "3-6 hrs" if s == "warning" else ("4-8 hrs" if s == "advisory" else "6-12 hrs"),
        "live_metrics": metrics,
        "meta": {
            "model_loaded":    True,
            "engine":          "GRU/LSTM Core Engine (Mock Demo Mode)",
            "target_location": "Low-Lying Catchments (Triangulo, Sabang, Mabolo), Naga City",
        },
    }


# ---------------------------------------------------------------------------
# GET /api/forecast
# ---------------------------------------------------------------------------
@app.get("/api/forecast")
def get_forecast():
    hourly = mock_hourly()
    daily  = mock_daily()

    next_6h  = sum(e["precipitation"] for e in hourly[:6])
    next_12h = sum(e["precipitation"] for e in hourly[:12])
    next_24h = sum(e["precipitation"] for e in hourly[:24])

    return {
        "status": "success",
        "location": "Barangay Triangulo, Naga City",
        "generated_at": datetime.datetime.now().strftime("%Y-%m-%dT%H:%M"),
        "outlook": {
            "next_6h_rain_mm":  round(next_6h,  2),
            "next_12h_rain_mm": round(next_12h, 2),
            "next_24h_rain_mm": round(next_24h, 2),
        },
        "hourly": hourly,
        "daily":  daily,
    }


# ---------------------------------------------------------------------------
# GET /api/pagasa-bulletin
# ---------------------------------------------------------------------------
@app.get("/api/pagasa-bulletin")
def get_pagasa_bulletin():
    f = cycle_factor()
    s = current_scenario()
    now_str  = datetime.datetime.now().strftime("%I:%M %p, %B %d %Y")
    next_str = (datetime.datetime.now() + datetime.timedelta(hours=3)).strftime("%I:%M %p")

    if s == "warning":
        rain_rate  = f"{lerp(10.0, 28.0, f):.0f}–{lerp(20.0, 38.0, f):.0f} mm/hr"
        wind_range = f"{lerp(55.0, 68.0, f):.0f}–{lerp(70.0, 85.0, f):.0f} kph"
        bulletin = (
            f"WEATHER BULLETIN (MOCK)\n"
            f"Issued by: PAGASA — Bicol Region\n"
            f"As of: {now_str}\n\n"
            f"A low pressure area associated with the ITCZ continues to bring moderate to heavy rainfall "
            f"over Camarines Sur and surrounding areas. Residents in low-lying and flood-prone areas, "
            f"including Barangay Triangulo in Naga City, are advised to take precautionary measures.\n\n"
            f"Rainfall: Moderate to heavy ({rain_rate} expected in the next 6 hours)\n"
            f"Wind: ENE at {wind_range}, gusts up to {lerp(80.0, 95.0, f):.0f} kph\n"
            f"Seas: Rough to very rough\n\n"
            f"Next bulletin at {next_str}."
        )
    elif s == "advisory":
        rain_rate  = f"{lerp(4.0, 10.0, f):.0f}–{lerp(8.0, 15.0, f):.0f} mm/hr"
        wind_range = f"{lerp(35.0, 48.0, f):.0f}–{lerp(50.0, 62.0, f):.0f} kph"
        bulletin = (
            f"WEATHER BULLETIN (MOCK)\n"
            f"Issued by: PAGASA — Bicol Region\n"
            f"As of: {now_str}\n\n"
            f"Scattered rainshowers and thunderstorms are expected over Camarines Sur due to the "
            f"enhanced Southwest Monsoon. Residents are advised to monitor weather updates and take "
            f"precautionary measures especially in low-lying and flood-prone areas.\n\n"
            f"Rainfall: Light to moderate ({rain_rate} expected in the next 6 hours)\n"
            f"Wind: SW at {wind_range}, gusts up to {lerp(60.0, 72.0, f):.0f} kph\n"
            f"Seas: Moderate to rough\n\n"
            f"Next bulletin at {next_str}."
        )
    else:  # normal
        rain_rate  = f"{lerp(1.0, 4.0, f):.0f}–{lerp(3.0, 6.0, f):.0f} mm/hr"
        wind_range = f"{lerp(8.0, 14.0, f):.0f}–{lerp(12.0, 18.0, f):.0f} kph"
        bulletin = (
            f"WEATHER BULLETIN (MOCK)\n"
            f"Issued by: PAGASA — Bicol Region\n"
            f"As of: {now_str}\n\n"
            f"Fair weather conditions prevail over Camarines Sur. "
            f"Isolated light rains may be experienced in the afternoon due to localized thunderstorms. "
            f"No significant weather disturbance affecting the area.\n\n"
            f"Rainfall: Light ({rain_rate})\n"
            f"Wind: NE at {wind_range} kph\n"
            f"Seas: Slight to moderate\n\n"
            f"Next bulletin at {next_str}."
        )

    return {
        "status": "success",
        "bulletin": bulletin,
        "source": "https://www1.pagasa.dost.gov.ph (mock)",
    }


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    s = current_scenario()
    return {
        "status": "ok",
        "remote_model_reachable": True,
        "model_loaded": True,
        "mode": f"MOCK — current scenario: {s.upper()}",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=False)