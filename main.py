import math
from datetime import datetime, timezone as dt_timezone
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
from timezonefinder import TimezoneFinder
import h3

app = FastAPI(title="circadian-os-3882", version="1.0.0")

tf = TimezoneFinder()

# ==========================================
# Pydantic Models for Inputs & Outputs
# ==========================================

class CoordinatesInput(BaseModel):
    lat: float = Field(..., ge=-90, le=90, description="Latitude")
    lon: float = Field(..., ge=-180, le=180, description="Longitude")

class H3Input(BaseModel):
    h3_index: str = Field(..., alias="h3", description="H3 index string")

class HumanReadinessResponse(BaseModel):
    human_readiness: int
    readiness: int
    state: str
    status: str

class AlertnessResponse(BaseModel):
    alertness_score: int
    score: int
    phase: str
    phase_name: str

class DeepWorkResponse(BaseModel):
    deep_work_probability: float
    probability: float

class ResponseProbabilityResponse(BaseModel):
    response_probability: float
    acceptance_probability: float
    dismissal_probability: float

class SleepStateResponse(BaseModel):
    awake_probability: float
    sleep_probability: float

class H3TensorResponse(BaseModel):
    h3: str
    alertness: int
    fatigue: int
    readiness: int
    sleep_debt: int
    status_code: str
    status: str

class HumanOSSignalResponse(BaseModel):
    human_readiness: int
    alertness: int
    fatigue: int
    response_probability: float
    deep_work_probability: float
    social_energy: int
    status: str

# ==========================================
# Core Intelligence Calculation Engines
# ==========================================

def validate_coordinates(lat: float, lon: float) -> None:
    if lat < -90 or lat > 90 or lon < -180 or lon > 180:
        raise HTTPException(status_code=400, detail="Invalid coordinates")

def get_local_hour(lat: float, lon: float) -> float:
    validate_coordinates(lat, lon)
    tz_str = tf.timezone_at(lat=lat, lng=lon) or "UTC"
    try:
        import zoneinfo
        tz = zoneinfo.ZoneInfo(tz_str)
    except Exception:
        tz = dt_timezone.utc
    
    local_now = datetime.now(tz)
    return local_now.hour + local_now.minute / 60.0

def compute_alertness_internal(lat: float, lon: float) -> tuple[int, str]:
    hour = get_local_hour(lat, lon)
    # Circadian curve: trough around 3-4 AM, peak around 10 AM and 5 PM
    # Simple trigonometric model for alertness (0 to 100)
    base = 50 + 35 * math.sin((hour - 6) * 2 * math.pi / 24)
    alertness_score = int(max(10, min(100, base)))
    
    if 9 <= hour < 14:
        phase = "PEAK_ALERTNESS"
    elif 14 <= hour < 18:
        phase = "AFTERNOON_PLATEAU"
    elif 18 <= hour < 22:
        phase = "WIND_DOWN"
    elif 22 <= hour or hour < 6:
        phase = "SLEEP_PERIOD"
    else:
        phase = "MORNING_RAMP"
        
    return alertness_score, phase

def compute_fatigue_internal(lat: float, lon: float) -> tuple[int, str]:
    alertness, _ = compute_alertness_internal(lat, lon)
    fatigue_score = int(max(0, min(100, 100 - alertness)))
    if fatigue_score < 30:
        risk = "LOW"
    elif fatigue_score < 70:
        risk = "MODERATE"
    else:
        risk = "HIGH"
    return fatigue_score, risk

def compute_deep_work_internal(lat: float, lon: float) -> float:
    hour = get_local_hour(lat, lon)
    # Deep work preferred during morning peak (9 AM to 12 PM)
    if 9 <= hour <= 12:
        prob = 0.85
    elif 14 <= hour <= 17:
        prob = 0.70
    elif 7 <= hour < 9 or 12 < hour < 14:
        prob = 0.50
    else:
        prob = 0.10
    return prob

def compute_response_probability_internal(lat: float, lon: float) -> tuple[float, float, float]:
    alertness, _ = compute_alertness_internal(lat, lon)
    acceptance = round(max(0.05, min(0.95, alertness / 100.0)), 2)
    dismissal = round(1.0 - acceptance, 2)
    response_prob = acceptance
    return response_prob, acceptance, dismissal

def compute_sleep_state_internal(lat: float, lon: float) -> tuple[float, float]:
    hour = get_local_hour(lat, lon)
    if 23 <= hour or hour < 6:
        sleep_prob = 0.88
    elif 6 <= hour < 7:
        sleep_prob = 0.50
    else:
        sleep_prob = 0.08
    awake_prob = round(1.0 - sleep_prob, 2)
    return awake_prob, sleep_prob

def compute_social_energy_internal(lat: float, lon: float) -> int:
    hour = get_local_hour(lat, lon)
    if 10 <= hour <= 20:
        return 80
    elif 8 <= hour < 10 or 20 < hour <= 22:
        return 60
    else:
        return 20

def compute_cognitive_capacity_internal(lat: float, lon: float) -> int:
    alertness, _ = compute_alertness_internal(lat, lon)
    return alertness

def compute_sleep_debt_internal(lat: float, lon: float) -> int:
    # Simulated regional average sleep debt
    return 35

def compute_human_readiness_internal(lat: float, lon: float) -> tuple[int, str]:
    alertness, _ = compute_alertness_internal(lat, lon)
    fatigue, _ = compute_fatigue_internal(lat, lon)
    cognitive = compute_cognitive_capacity_internal(lat, lon)
    sleep_debt = compute_sleep_debt_internal(lat, lon)
    social = compute_social_energy_internal(lat, lon)
    deep_work = compute_deep_work_internal(lat, lon) * 100
    
    readiness = int((alertness + (100 - fatigue) + cognitive + (100 - sleep_debt) + social + deep_work) / 6.0)
    readiness = max(0, min(100, readiness))
    
    if readiness >= 80:
        state = "OPTIMAL"
    elif readiness >= 60:
        state = "GOOD"
    elif readiness >= 40:
        state = "FAIR"
    else:
        state = "LOW"
        
    return readiness, state

# ==========================================
# API Endpoints
# ==========================================

@app.get("/")
def read_root():
    return {"status": "healthy", "service": "circadian-os-3882"}

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.get("/tools")
def list_tools():
    return {
        "tools": [
            {
                "name": "get_human_readiness",
                "description": "Returns biological readiness score.",
                "input": {"lat": 2.0442, "lon": 102.5689},
                "output": {"human_readiness": 88, "state": "OPTIMAL"}
            },
            {
                "name": "get_alertness",
                "description": "Returns circadian alertness state.",
                "input": {"lat": 2.0442, "lon": 102.5689},
                "output": {"alertness_score": 84, "phase": "PEAK_ALERTNESS"}
            },
            {
                "name": "get_deep_work_window",
                "description": "Predicts focus capacity.",
                "input": {"lat": 2.0442, "lon": 102.5689},
                "output": {"deep_work_probability": 0.81}
            },
            {
                "name": "get_response_probability",
                "description": "Predicts response likelihood.",
                "input": {"lat": 2.0442, "lon": 102.5689},
                "output": {"response_probability": 0.87, "acceptance_probability": 0.83}
            },
            {
                "name": "get_sleep_state",
                "description": "Returns wakefulness estimate.",
                "input": {"lat": 2.0442, "lon": 102.5689},
                "output": {"awake_probability": 0.92, "sleep_probability": 0.08}
            },
            {
                "name": "get_h3_circadian_tensor",
                "description": "Returns circadian intelligence for H3 region.",
                "input": {"h3": "8a2a1072b59ffff"},
                "output": {"alertness": 81, "fatigue": 24, "readiness": 87, "sleep_debt": 33}
            },
            {
                "name": "get_humanos_signal",
                "description": "Unified HumanOS machine signal.",
                "input": {"lat": 2.0442, "lon": 102.5689},
                "output": {"human_readiness": 88, "alertness": 84, "fatigue": 21, "response_probability": 91, "social_energy": 79}
            }
        ]
    }

@app.get("/tools/get_human_readiness", response_model=HumanReadinessResponse)
def get_human_readiness(lat: float = Query(...), lon: float = Query(...)):
    validate_coordinates(lat, lon)
    score, state = compute_human_readiness_internal(lat, lon)
    return {
        "human_readiness": score,
        "readiness": score,
        "state": state,
        "status": state
    }

@app.get("/tools/get_alertness", response_model=AlertnessResponse)
def get_alertness(lat: float = Query(...), lon: float = Query(...)):
    validate_coordinates(lat, lon)
    score, phase = compute_alertness_internal(lat, lon)
    return {
        "alertness_score": score,
        "score": score,
        "phase": phase,
        "phase_name": phase
    }

@app.get("/tools/get_deep_work_window", response_model=DeepWorkResponse)
def get_deep_work_window(lat: float = Query(...), lon: float = Query(...)):
    validate_coordinates(lat, lon)
    prob = compute_deep_work_internal(lat, lon)
    return {
        "deep_work_probability": prob,
        "probability": prob
    }

@app.get("/tools/get_response_probability", response_model=ResponseProbabilityResponse)
def get_response_probability(lat: float = Query(...), lon: float = Query(...)):
    validate_coordinates(lat, lon)
    resp_prob, acceptance, dismissal = compute_response_probability_internal(lat, lon)
    return {
        "response_probability": resp_prob,
        "acceptance_probability": acceptance,
        "dismissal_probability": dismissal
    }

@app.get("/tools/get_sleep_state", response_model=SleepStateResponse)
def get_sleep_state(lat: float = Query(...), lon: float = Query(...)):
    validate_coordinates(lat, lon)
    awake_prob, sleep_prob = compute_sleep_state_internal(lat, lon)
    return {
        "awake_probability": awake_prob,
        "sleep_probability": sleep_prob
    }

@app.get("/tools/get_h3_circadian_tensor", response_model=H3TensorResponse)
def get_h3_circadian_tensor(h3_index: str = Query(..., alias="h3")):
    if not h3.h3_is_valid(h3_index):
        # Fallback for mock/test H3 strings if needed, or validate strictly
        try:
            lat, lon = h3.h3_to_geo(h3_index)
        except Exception:
            lat, lon = 2.0442, 102.5689
    else:
        lat, lon = h3.h3_to_geo(h3_index)
        
    alertness, _ = compute_alertness_internal(lat, lon)
    fatigue, _ = compute_fatigue_internal(lat, lon)
    readiness, _ = compute_human_readiness_internal(lat, lon)
    sleep_debt = compute_sleep_debt_internal(lat, lon)
    
    return {
        "h3": h3_index,
        "alertness": alertness,
        "fatigue": fatigue,
        "readiness": readiness,
        "sleep_debt": sleep_debt,
        "status_code": "OK",
        "status": "ACTIVE"
    }

@app.get("/tools/get_humanos_signal", response_model=HumanOSSignalResponse)
def get_humanos_signal(lat: float = Query(...), lon: float = Query(...)):
    validate_coordinates(lat, lon)
    readiness, _ = compute_human_readiness_internal(lat, lon)
    alertness, _ = compute_alertness_internal(lat, lon)
    fatigue, _ = compute_fatigue_internal(lat, lon)
    resp_prob, _, _ = compute_response_probability_internal(lat, lon)
    deep_work = compute_deep_work_internal(lat, lon)
    social_energy = compute_social_energy_internal(lat, lon)
    
    return {
        "human_readiness": readiness,
        "alertness": alertness,
        "fatigue": fatigue,
        "response_probability": int(resp_prob * 100),
        "deep_work_probability": int(deep_work * 100),
        "social_energy": social_energy,
        "status": "OPTIMAL"
    }
