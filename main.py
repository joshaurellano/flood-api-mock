"""
AGOS Mock Flood API — Demo / Defense-Presentation Mode
========================================================

Drop-in replacement for the real backend (backend.zip / app/) during a
live demo. Every endpoint here returns the SAME field names/shapes as the
real FastAPI backend (see app/api/routes_flood.py, routes_weather.py,
app/models/inference.py) so the existing frontend renders identically —
only the underlying numbers are scripted instead of coming from a real
model + Open-Meteo.

Point VITE_LIVE_URL (or whatever env var your frontend's apiBaseUrl reads)
at wherever this is deployed, same as the real backend.

── THE DEMO NARRATIVE ──────────────────────────────────────────────────
Time repeats in a CYCLE_SECONDS loop (default 5 minutes) so you can just
let it run in the background and it'll always be a few minutes from the
next "interesting" moment. Five phases:

  0. CALM            (0:00-1:00)  Today's reading is NORMAL. Forecast is
                                   flat/low across all 14 days too.
  1. BREWING         (1:00-2:00)  Today STILL reads NORMAL -- but the
                                   3-day-ahead forecast (day_ahead 1-2)
                                   climbs into WARNING/CRITICAL. This is
                                   the moment to point at the 3-day
                                   forecast alert (FORECAST_WARNING) --
                                   current conditions look fine, the
                                   forecast doesn't.
  2. RISING          (2:00-3:00)  Today's OWN probability now climbs
                                   quickly, crossing ADVISORY -> WARNING ->
                                   CRITICAL -- this is the rate-of-rise /
                                   state-change moment.
  3. SUSTAINED        (3:00-4:00) Holds at CRITICAL/WARNING -- the
                                   "still elevated" reminder window.
  4. RECOVERING       (4:00-5:00) Drops back toward NORMAL, but the
                                   forecast still shows residual risk a
                                   day or two out -- the downgrade-caution
                                   moment ("back to normal, BUT...").

Then it loops back to CALM.

── IMPORTANT TIMING CAVEAT -- READ BEFORE YOUR DEMO ────────────────────
poll-flood and check-forecast (the Edge Functions) use real-world
windows (a 2-hour rate-of-rise lookback, a 3-hour debounce/reminder
interval, a several-hour forecast-check cron). This mock's 5-minute
cycle is fast enough to LOOK dynamic on screen, but too fast for those
Edge Functions' own cron schedules to necessarily catch every phase
transition and fire an alert live during a short demo window. Either:
  (a) temporarily shorten poll-flood's RAPID_RISE_WINDOW_MINUTES /
      STILL_ELEVATED_REMINDER_HOURS and check-forecast's cron interval
      before the presentation, or
  (b) manually invoke poll-flood / check-forecast (a plain curl/Postman
      call to their URLs) at the right moment in your talk instead of
      waiting on their cron schedules, or
  (c) use the phase-locking feature below to hold the mock in one
      phase indefinitely so you (or the Edge Function's cron) don't
      have to race the 5-minute loop at all.
Either way, the mock's job is just to make predict-flood/forecast-flood
LOOK right when the frontend (or your manually-triggered Edge Function)
reads it -- it doesn't control when those functions decide to look.

── PHASE LOCKING (new) — for testing the 3-day-forecast alert ─────────
By default the mock free-runs through the 5-phase cycle above, so the
"BREWING" window (today NORMAL, day_ahead 1-2 WARNING/CRITICAL) — the
exact scenario the 3-day-forecast alert is meant to catch — only exists
for 1/5th of the cycle and then moves on. That's a moving target if
you're trying to confirm the Edge Function's 3-day-crossing logic
actually fires.

You can now pin the mock to one phase and leave it there:

  1. Per-request (no redeploy needed) — add a `phase` query param to
     any endpoint:
         GET /api/forecast-flood?phase=brewing
         GET /api/predict-flood?phase=brewing
         GET /api/forecast?phase=brewing
     Accepts either the phase name (calm, brewing, rising, sustained,
     recovering — case-insensitive) or its index (0-4). While locked,
     the response still animates smoothly WITHIN that phase (looping
     every phase_len = CYCLE_SECONDS/5 seconds) — it isn't a single
     frozen snapshot, so repeated polls still look "live" and today
     stays NORMAL / day_ahead 1-2 stay WARNING+ the whole time you're
     testing.

  2. Whole-deployment default — set the env var MOCK_FORCE_PHASE
     (e.g. `MOCK_FORCE_PHASE=brewing` or `MOCK_FORCE_PHASE=1`) before
     starting the server, so every request is locked to that phase
     unless a request overrides it with its own `?phase=` param. Unset
     (default) = free-running 5-phase cycle as before.

  3. To go back to the normal free-running cycle for a request, either
     don't pass `phase`, or pass `phase=cycle` / `phase=auto`.

This only changes WHICH phase is being read; it doesn't change any of
the underlying probability curves, field shapes, or thresholds, so the
frontend/Edge Functions still see exactly the same kind of payload they
would during a normal free-running demo — just held on the moment you
want to test against instead of cycling past it.
──────────────────────────────────────────────────────────────────────
"""

import contextvars
import datetime
import os
import time
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

try:
    from dotenv import load_dotenv
    # Looks for a .env file in the current working directory (or wherever
    # you point it with load_dotenv(dotenv_path=...)) and copies its
    # key=value lines into os.environ *before* we read MOCK_FORCE_PHASE /
    # MOCK_CYCLE_SECONDS below. Safe no-op if no .env file exists.
    load_dotenv()
except ImportError:
    # python-dotenv not installed in this environment -- env vars set the
    # normal way (shell export, docker-compose, host platform config, etc.)
    # still work fine; only a .env file specifically would be ignored.
    pass

app = FastAPI(
    title="AGOS Mock Flood API — Demo Mode",
    description="Scripted, deterministic stand-in for the real flood-prediction backend, for live demos/presentations.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Tunables -- override via env var without touching code, e.g. to speed
# the whole cycle up for a rehearsal or slow it down for a long defense.
# ---------------------------------------------------------------------------
CYCLE_SECONDS = int(os.environ.get("MOCK_CYCLE_SECONDS", "300"))  # 5 min default
LOCATION = "Barangay Triangulo, Naga City"
COORDS = (13.6192, 123.1814)  # same as the real backend's settings.LAT/LON

MODEL_REGISTRY = {
    "gru":  {"label": "GRU Encoder-Decoder"},
    "lstm": {"label": "LSTM Encoder-Decoder"},
    "cnn":  {"label": "CNN Encoder-Decoder"},
}
DEFAULT_MODEL_KEY = "gru"

PHASE_NAMES = ["calm", "brewing", "rising", "sustained", "recovering"]
PHASE_DISPLAY_NAMES = [
    "CALM",
    "BREWING (3-day forecast rising)",
    "RISING (today climbing)",
    "SUSTAINED (still elevated)",
    "RECOVERING (downgrade w/ residual risk)",
]


def _resolve_phase_token(value) -> Optional[int]:
    """Turn 'brewing' / 'BREWING' / '1' / 1 into a phase index 0-4, or
    None if the value doesn't map to a phase (e.g. 'cycle'/'auto'/empty)."""
    if value is None:
        return None
    token = str(value).strip().lower()
    if token in ("", "cycle", "auto", "none"):
        return None
    if token in PHASE_NAMES:
        return PHASE_NAMES.index(token)
    try:
        idx = int(token)
    except ValueError:
        return None
    return idx if 0 <= idx <= 4 else None


# Deployment-wide default lock, e.g. MOCK_FORCE_PHASE=brewing
_ENV_FORCE_PHASE = _resolve_phase_token(os.environ.get("MOCK_FORCE_PHASE"))

# Per-request override (set at the top of each endpoint from the `phase`
# query param). Falls back to _ENV_FORCE_PHASE when unset for a request.
_forced_phase_ctx: "contextvars.ContextVar[Optional[int]]" = contextvars.ContextVar(
    "forced_phase", default=None
)


def _apply_phase_override(phase: Optional[str]) -> None:
    """Call at the top of every endpoint with its `phase` query param."""
    override = _resolve_phase_token(phase)
    _forced_phase_ctx.set(override if override is not None else _ENV_FORCE_PHASE)


def _now():
    return datetime.datetime.now()


def _elapsed() -> float:
    """Seconds into the current cycle, 0 <= t < CYCLE_SECONDS."""
    return time.time() % CYCLE_SECONDS


def _phase_fraction(elapsed: float) -> tuple:
    """Returns (phase_index 0-4, progress 0.0-1.0 within that phase).

    If a phase is locked (via ?phase= query param or MOCK_FORCE_PHASE
    env var), phase_index is pinned to that value and progress simply
    loops within that one phase's window every phase_len seconds, so
    values keep animating instead of freezing on one exact number.
    """
    phase_len = CYCLE_SECONDS / 5.0
    forced = _forced_phase_ctx.get()
    if forced is not None:
        progress = (elapsed % phase_len) / phase_len
        return forced, progress
    phase = min(int(elapsed // phase_len), 4)
    progress = (elapsed - phase * phase_len) / phase_len
    return phase, max(0.0, min(1.0, progress))


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def probability_to_alert_level(p: float) -> str:
    """Identical thresholds to the real backend's app/utils/alerts.py."""
    if p >= 0.75:
        return "CRITICAL"
    if p >= 0.50:
        return "WARNING"
    if p >= 0.25:
        return "ADVISORY"
    return "NORMAL"


# ---------------------------------------------------------------------------
# The scripted probability curves -- this is the heart of the demo.
# today_probability(): day_ahead = 0, i.e. "right now".
# horizon_probability(day_ahead, elapsed): everything else (1..13 days out).
# ---------------------------------------------------------------------------
def today_probability(elapsed: float) -> float:
    phase, t = _phase_fraction(elapsed)

    if phase == 0:      # CALM
        return round(lerp(0.05, 0.12, t), 4)
    elif phase == 1:    # BREWING -- today still reads calm on purpose
        return round(lerp(0.10, 0.16, t), 4)
    elif phase == 2:    # RISING -- the rapid-rise moment
        return round(lerp(0.16, 0.85, t ** 0.7), 4)  # convex: slow start, fast climb
    elif phase == 3:    # SUSTAINED
        return round(lerp(0.85, 0.80, t) + 0.03 * (1 if int(elapsed) % 2 == 0 else 0), 4)
    else:                # RECOVERING
        return round(lerp(0.78, 0.10, t), 4)


def horizon_probability(day_ahead: int, elapsed: float) -> float:
    """
    day_ahead = 0 is handled by today_probability() directly (kept in sync
    below). This covers day_ahead 1..13 for the 14-day forecast array.
    """
    phase, t = _phase_fraction(elapsed)
    today_p = today_probability(elapsed)

    if day_ahead == 0:
        return today_p

    if phase in (0, 1):
        # CALM / BREWING: the whole point is days 1-2 spike ahead of today.
        # day_ahead 1 ramps partway, day_ahead 2 peaks into WARNING/CRITICAL.
        peak_by_day = {1: 0.55, 2: 0.78}
        if day_ahead in peak_by_day:
            target = peak_by_day[day_ahead]
            # Ramp the spike in smoothly over CALM, hold/intensify through BREWING.
            ramp = t if phase == 0 else 1.0
            extra = 0.15 if phase == 1 else 0.0  # BREWING intensifies further
            return round(min(0.97, lerp(today_p, target + extra, ramp)), 4)
        # days 3+ taper back down from the day_ahead=2 peak toward a mild baseline
        decay = max(0.0, 1 - (day_ahead - 2) * 0.12)
        return round(max(0.05, 0.55 * decay + today_p * (1 - decay)), 4)

    if phase == 2:
        # RISING: the forecast "catches up" to today's own climb -- days 1-3
        # converge toward today_p, further days stay mildly elevated.
        converge = min(1.0, day_ahead / 3.0)
        far_baseline = 0.35
        return round(lerp(today_p, far_baseline, converge * 0.6), 4)

    if phase == 3:
        # SUSTAINED: everything nearby stays elevated, decaying gently with distance.
        decay = max(0.15, 1 - day_ahead * 0.06)
        return round(today_p * decay, 4)

    # RECOVERING: today drops, but day_ahead 1-2 lag behind on purpose --
    # this is what powers the "forecast still shows risk" downgrade message.
    lag_target = {1: 0.42, 2: 0.30}
    if day_ahead in lag_target:
        return round(max(today_p, lerp(0.80, lag_target[day_ahead], t)), 4)
    decay = max(0.1, 1 - (day_ahead - 2) * 0.08)
    return round(max(0.05, today_p * decay + 0.05), 4)


def build_forecast_probs(elapsed: float, horizon: int = 14) -> list:
    return [
        (today_probability(elapsed) if d == 0 else horizon_probability(d, elapsed))
        for d in range(horizon)
    ]


# ---------------------------------------------------------------------------
# Live metrics -- same field names as app/features/aggregation.py get_live_metrics()
# ---------------------------------------------------------------------------
def live_metrics(elapsed: float) -> dict:
    p = today_probability(elapsed)
    level = probability_to_alert_level(p)

    if level == "CRITICAL":
        rainfall, wind, humidity = lerp(22.0, 32.0, p), lerp(70.0, 88.0, p), lerp(93.0, 98.0, p)
    elif level == "WARNING":
        rainfall, wind, humidity = lerp(12.0, 22.0, p), lerp(55.0, 70.0, p), lerp(88.0, 93.0, p)
    elif level == "ADVISORY":
        rainfall, wind, humidity = lerp(4.0, 12.0, p), lerp(38.0, 55.0, p), lerp(80.0, 88.0, p)
    else:
        rainfall, wind, humidity = lerp(0.0, 2.0, p * 4), lerp(8.0, 20.0, p * 4), lerp(68.0, 80.0, p * 4)

    return {
        "rainfall_mm": round(rainfall, 2),
        "humidity": round(humidity, 1),
        "wind_signal": 2 if wind >= 62 else (1 if wind >= 39 else 0),
        "max_wind_kph": round(wind, 1),
        "wind_direction_deg": round(70 + 40 * p, 1),
        "pressure_msl_hpa": round(1011 - 12 * p, 1),
        "surface_pressure_hpa": round(1008 - 12 * p, 1),
        "wind_gusts_kph": round(wind * 1.3, 1),
        "soil_moisture_vwc": round(0.28 + 0.35 * p, 4),
        "feels_like_c": round(29 - 3 * p, 1),
        "is_day": 6 <= _now().hour < 18,
    }


# ---------------------------------------------------------------------------
# weather_cache / pagasa_calibration -- same shape as the real diagnostic
# blocks (app/weather/cache.py, app/weather/calibration.py), always
# reporting a healthy "fresh" state since the mock has nothing to fetch.
# ---------------------------------------------------------------------------
def mock_weather_cache() -> dict:
    return {
        "status": "fresh",
        "age_minutes": 0,
        "last_successful_fetch": _now().isoformat(timespec="seconds"),
        "significantly_stale": False,
        "loaded_from_disk": False,
        "fallback_source": None,
        "in_failure_cooldown": False,
    }


def mock_pagasa_calibration() -> dict:
    fields = ["relative_humidity_2m", "wind_speed_10m", "pressure_msl", "precipitation"]
    return {
        "enabled": True,
        "precipitation_calibration_enabled": False,
        "reference_station": {"id": "5037", "name": "Pili, Camarines Sur AWS"},
        "total_samples": 500,
        "min_samples_required": 20,
        "min_samples_required_precipitation": 50,
        "max_samples_kept": 500,
        "per_field": {
            f: {"samples": 500, "bias": 0.0, "applied": f != "precipitation", "unit": "n/a"}
            for f in fields
        },
        "notes": "Mock demo mode -- calibration values are static placeholders, not live PAGASA data.",
        "last_sample": None,
    }


# ---------------------------------------------------------------------------
# meta block -- same shape/keys as app/models/inference.py _meta_block()
# ---------------------------------------------------------------------------
def meta_block(model_key: str) -> dict:
    info = MODEL_REGISTRY[model_key]
    return {
        "model_key": model_key,
        "engine": f"{info['label']} 14-Day Model (MOCK — scripted demo data, not a live forward pass)",
        "note": (
            "DEMO MODE: this response is scripted for a presentation, not computed by a real model. "
            "Field shapes match the production API exactly so the frontend behaves identically."
        ),
        "assumptions": [
            "Decoder input (rain/wind/typhoon-signal features for the next 14 days) comes directly from Open-Meteo's forecast, not a blind extrapolation of past patterns.",
            "Antecedent moisture is a proxy derived from computed rainfall API. No live soil-moisture sensor.",
            "prev_flood defaults to 0 in the encoder because there is no live ground-truth flood feed. Excluded entirely from the decoder.",
            "Typhoon signal is a proxy derived from forecasted wind-speed thresholds.",
        ],
        "model_reliability": {
            "avg_precision": 0.87,
            "avg_recall": 0.83,
            "avg_f1": 0.85,
            "avg_false_alarm_rate": 0.09,
            "measured_on": "Held-out test split (mock -- see the real backend's per_horizon_metrics.csv for actual figures)",
            "plain_language": "Treat flood_probability as a decision-support signal, not a certainty score.",
        },
        "enriched_metrics_used_by_model": [
            "soil_moisture_vwc", "pressure_msl_hpa", "surface_pressure_hpa", "wind_gusts_kph",
        ],
        "enriched_metrics_note": (
            "soil_moisture_vwc / pressure_msl_hpa / surface_pressure_hpa / wind_gusts_kph are fetched "
            "from Open-Meteo in production. They only feed the model when listed above."
        ),
    }


def _build_forecast_entries(elapsed: float) -> list:
    """Same per-day shape as inference.py _build_forecast_entries()."""
    probs = build_forecast_probs(elapsed)
    today = datetime.date.today()
    out = []
    for day_ahead, p in enumerate(probs):
        forecast_date = today + datetime.timedelta(days=day_ahead)
        out.append({
            "date": forecast_date.isoformat(),
            "day_ahead": day_ahead,
            "flood_probability": round(p, 4),
            "alert_level": probability_to_alert_level(p),
            "confidence_band": "high" if day_ahead < 3 else ("moderate" if day_ahead < 7 else "outlook-only"),
            "soil_moisture_vwc": round(0.28 + 0.35 * p, 4),
            "pressure_msl_hpa": round(1011 - 12 * p, 1),
            "surface_pressure_hpa": round(1008 - 12 * p, 1),
            "wind_gusts_kph": round((20 + 70 * p) * 1.3, 1),
            "rainfall_mm": round(35 * p, 1),
            "wind_speed_max_kph": round(20 + 70 * p, 1),
        })
    return out


def _forecast_response(model_key: str) -> dict:
    elapsed = _elapsed()
    return {
        "status": "success",
        "generated_at": _now().isoformat(timespec="seconds"),
        "weather_cache": mock_weather_cache(),
        "pagasa_calibration": mock_pagasa_calibration(),
        "model_input_past_dates": [
            (datetime.date.today() - datetime.timedelta(days=d)).isoformat() for d in range(16, 0, -1)
        ],
        "model_input_forecast_dates": [
            (datetime.date.today() + datetime.timedelta(days=d)).isoformat() for d in range(16)
        ],
        "live_metrics": live_metrics(elapsed),
        "meta": meta_block(model_key),
        "forecast": _build_forecast_entries(elapsed),
    }


# ===========================================================================
# GET /api/forecast-flood/{gru,lstm,cnn} + default + compare
# ===========================================================================
@app.get("/api/forecast-flood/gru")
def forecast_flood_gru(phase: Optional[str] = Query(default=None)):
    _apply_phase_override(phase)
    return _forecast_response("gru")


@app.get("/api/forecast-flood/lstm")
def forecast_flood_lstm(phase: Optional[str] = Query(default=None)):
    _apply_phase_override(phase)
    return _forecast_response("lstm")


@app.get("/api/forecast-flood/cnn")
def forecast_flood_cnn(phase: Optional[str] = Query(default=None)):
    _apply_phase_override(phase)
    return _forecast_response("cnn")


@app.get("/api/forecast-flood/compare")
def forecast_flood_compare(phase: Optional[str] = Query(default=None)):
    _apply_phase_override(phase)
    elapsed = _elapsed()
    keys = list(MODEL_REGISTRY.keys())
    per_model = {k: {"label": MODEL_REGISTRY[k]["label"], "meta": meta_block(k), "forecast": _build_forecast_entries(elapsed)} for k in keys}

    probs = build_forecast_probs(elapsed)
    today = datetime.date.today()
    comparison = []
    for day_ahead, p in enumerate(probs):
        forecast_date = today + datetime.timedelta(days=day_ahead)
        # All three "models" agree in this mock (same scripted curve) -- real
        # backend would show genuine per-algorithm spread here.
        day_probs = {k: round(p, 4) for k in keys}
        alert_levels = {k: probability_to_alert_level(p) for k in keys}
        comparison.append({
            "date": forecast_date.isoformat(),
            "day_ahead": day_ahead,
            "probabilities": day_probs,
            "alert_levels": alert_levels,
            "ensemble_mean_probability": round(p, 4),
            "ensemble_alert_level": probability_to_alert_level(p),
            "spread": 0.0,
            "models_agree": True,
        })

    return {
        "status": "success",
        "generated_at": _now().isoformat(timespec="seconds"),
        "weather_cache": mock_weather_cache(),
        "pagasa_calibration": mock_pagasa_calibration(),
        "model_input_past_dates": [(today - datetime.timedelta(days=d)).isoformat() for d in range(16, 0, -1)],
        "model_input_forecast_dates": [(today + datetime.timedelta(days=d)).isoformat() for d in range(16)],
        "live_metrics": live_metrics(elapsed),
        "models_compared": keys,
        "default_model": DEFAULT_MODEL_KEY,
        "per_model": per_model,
        "comparison": comparison,
    }


@app.get("/api/forecast-flood")
def forecast_flood_default(phase: Optional[str] = Query(default=None)):
    _apply_phase_override(phase)
    return _forecast_response(DEFAULT_MODEL_KEY)


# ===========================================================================
# GET /api/predict-flood (+ per-model) -- day-1-only summary, same shape as
# app/api/routes_flood.py predict_flood() / predict_flood_for_model()
# ===========================================================================
def _predict_response(model_key: str) -> dict:
    full = _forecast_response(model_key)
    today_entry = full["forecast"][0]
    return {
        "status": "success",
        "model_key": model_key,
        "alert_level": today_entry["alert_level"],
        "probability": today_entry["flood_probability"],
        "live_metrics": full["live_metrics"],
        "weather_cache": full["weather_cache"],
        "pagasa_calibration": full["pagasa_calibration"],
        "meta": full["meta"],
    }


@app.get("/api/predict-flood")
def predict_flood(phase: Optional[str] = Query(default=None)):
    _apply_phase_override(phase)
    return _predict_response(DEFAULT_MODEL_KEY)


@app.get("/api/predict-flood/{model_key}")
def predict_flood_for_model(model_key: str, phase: Optional[str] = Query(default=None)):
    _apply_phase_override(phase)
    if model_key not in MODEL_REGISTRY:
        raise HTTPException(status_code=404, detail=f"Unknown model '{model_key}'. Valid options: {list(MODEL_REGISTRY)}")
    return _predict_response(model_key)


# ===========================================================================
# GET /api/forecast -- general 48h-hourly + 14-day-daily weather endpoint,
# same shape as app/api/routes_weather.py get_forecast()
# ===========================================================================
def _wmo_for_level(level: str) -> tuple:
    return {
        "CRITICAL": (65, "Heavy rain"),
        "WARNING":  (63, "Moderate rain"),
        "ADVISORY": (51, "Light drizzle"),
        "NORMAL":   (2,  "Partly cloudy"),
    }[level]


@app.get("/api/forecast")
def get_forecast(phase: Optional[str] = Query(default=None)):
    _apply_phase_override(phase)
    elapsed = _elapsed()
    now = _now().replace(minute=0, second=0, microsecond=0)
    probs = build_forecast_probs(elapsed)

    minutely = []
    for i in range(8):
        p = today_probability(elapsed)
        precip_15 = round((35 * p / 24 / 4), 2)
        minutely.append({
            "time": (now + datetime.timedelta(minutes=15 * i)).strftime("%Y-%m-%dT%H:%M"),
            "precipitation_mm": precip_15,
            "precipitation_rate_mmhr": round(precip_15 * 4, 2),
        })

    hourly = []
    for i in range(48):
        # Interpolate across whichever forecast day this hour falls in.
        day_idx = min(i // 24, len(probs) - 1)
        p = probs[day_idx]
        level = probability_to_alert_level(p)
        wmo, condition = _wmo_for_level(level)
        wind = round(20 + 70 * p + (i % 4) * 1.2, 1)
        hourly.append({
            "time": (now + datetime.timedelta(hours=i)).strftime("%Y-%m-%dT%H:%M"),
            "precipitation": round((35 * p / 24) + (i % 3) * 0.1, 2),
            "rain_probability_pct": round(min(100, p * 120)),
            "humidity": round(70 + 25 * p, 1),
            "wind_speed_kph": wind,
            "temperature_c": round(29 - 3 * p - (i % 6) * 0.2, 1),
            "feels_like_c": round(29 - 3 * p, 1),
            "is_day": 6 <= (now + datetime.timedelta(hours=i)).hour < 18,
            "visibility_km": round(24 - 15 * p, 1),
            "uv_index": round(max(0, 9 - 8 * p), 1),
            "dew_point_c": round(23 - 2 * p, 1),
            "weathercode": wmo,
            "condition": condition,
            "wind_signal": 2 if wind >= 62 else (1 if wind >= 39 else 0),
            "soil_moisture_vwc": round(0.28 + 0.35 * p, 4),
            "pressure_msl_hpa": round(1011 - 12 * p, 1),
            "surface_pressure_hpa": round(1008 - 12 * p, 1),
            "wind_gusts_kph": round(wind * 1.3, 1),
        })

    daily = []
    today_date = datetime.date.today()
    for day_ahead, p in enumerate(probs):
        level = probability_to_alert_level(p)
        wmo, condition = _wmo_for_level(level)
        wind_max = round(20 + 70 * p, 1)
        daily.append({
            "date": (today_date + datetime.timedelta(days=day_ahead)).isoformat(),
            "precipitation_sum_mm": round(35 * p, 1),
            "rain_probability_pct": round(min(100, p * 120)),
            "temperature_max_c": round(29 - 3 * p, 1),
            "wind_speed_max_kph": wind_max,
            "wind_signal": 2 if wind_max >= 62 else (1 if wind_max >= 39 else 0),
            "weathercode": wmo,
            "condition": condition,
            "soil_moisture_vwc": round(0.28 + 0.35 * p, 4),
            "pressure_msl_hpa": round(1011 - 12 * p, 1),
            "surface_pressure_hpa": round(1008 - 12 * p, 1),
            "wind_gusts_max_kph": round(wind_max * 1.3, 1),
        })

    next_6h = sum(e["precipitation"] for e in hourly[:6])
    next_12h = sum(e["precipitation"] for e in hourly[:12])
    next_24h = sum(e["precipitation"] for e in hourly[:24])

    def _max_prob(entries):
        vals = [e["rain_probability_pct"] for e in entries if e["rain_probability_pct"] is not None]
        return max(vals) if vals else None

    return {
        "status": "success",
        "location": LOCATION,
        "generated_at": _now().strftime("%Y-%m-%dT%H:%M"),
        "weather_cache": mock_weather_cache(),
        "pagasa_calibration": mock_pagasa_calibration(),
        "outlook": {
            "next_6h_rain_mm": round(next_6h, 2),
            "next_6h_rain_probability_pct": _max_prob(hourly[:6]),
            "next_12h_rain_mm": round(next_12h, 2),
            "next_12h_rain_probability_pct": _max_prob(hourly[:12]),
            "next_24h_rain_mm": round(next_24h, 2),
            "next_24h_rain_probability_pct": _max_prob(hourly[:24]),
        },
        "live_metrics": live_metrics(elapsed),
        "minutely": minutely,
        "hourly": hourly,
        "daily": daily,
    }


# ===========================================================================
# GET /health -- same shape as the real backend's health check, plus the
# current demo phase so you can glance at it and know where the cycle is
# ===========================================================================
@app.get("/health")
def health(phase: Optional[str] = Query(default=None)):
    _apply_phase_override(phase)
    elapsed = _elapsed()
    phase_idx, progress = _phase_fraction(elapsed)
    return {
        "status": "ok",
        "remote_model_reachable": True,
        "model_loaded": True,
        "mode": "MOCK — Demo/Defense Presentation Mode",
        "demo_phase": PHASE_DISPLAY_NAMES[phase_idx],
        "demo_phase_progress": round(progress, 2),
        "demo_cycle_seconds": CYCLE_SECONDS,
        "demo_seconds_elapsed_in_cycle": round(elapsed, 1),
        "demo_phase_locked": _forced_phase_ctx.get() is not None,
        "demo_phase_lock_source": (
            "query_param" if _resolve_phase_token(phase) is not None
            else ("env:MOCK_FORCE_PHASE" if _ENV_FORCE_PHASE is not None else None)
        ),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=False)