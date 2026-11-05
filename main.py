import math
from datetime import datetime, timezone as dt_timezone
from typing import Any, Dict, List, Optional, Tuple, Union
from enum import Enum

import numpy as np
from fastapi import FastAPI, HTTPException, Query, Body
from fastapi.responses import ORJSONResponse
from pydantic import BaseModel, Field
from timezonefinder import TimezoneFinder
import h3

# ---------------------------------------------------------------------------
# Framework & App Initialization
# ---------------------------------------------------------------------------
app = FastAPI(
    title="circadian-os-3882",
    description="Human Biological State, Circadian Alertness & Cognitive Readiness Intelligence Engine",
    version="2.0.0",
    default_response_class=ORJSONResponse,
)

tf = TimezoneFinder()

# ---------------------------------------------------------------------------
# Domain Enums
# ---------------------------------------------------------------------------
class Chronotype(str, Enum):
    EARLY_BIRD = "EARLY_BIRD"
    NEUTRAL = "NEUTRAL"
    NIGHT_OWL = "NIGHT_OWL"

class ReadinessClassification(str, Enum):
    ASLEEP = "ASLEEP"
    RECOVERING = "RECOVERING"
    ACTIVE = "ACTIVE"
    HIGH_PERFORMANCE = "HIGH_PERFORMANCE"
    PEAK_COGNITIVE_STATE = "PEAK_COGNITIVE_STATE"

class AlertnessPhase(str, Enum):
    PEAK_ALERTNESS = "PEAK_ALERTNESS"
    AFTERNOON_PLATEAU = "AFTERNOON_PLATEAU"
    EVENING_BOOST = "EVENING_BOOST"
    WIND_DOWN = "WIND_DOWN"
    SLEEP_PERIOD = "SLEEP_PERIOD"
    MORNING_RAMP = "MORNING_RAMP"

# ---------------------------------------------------------------------------
# Global Metro Coordinates Registry
# ---------------------------------------------------------------------------
GLOBAL_WORKFORCE_REGISTRY: Dict[str, Dict[str, Any]] = {
    "kuala lumpur": {"city": "Kuala Lumpur", "country": "Malaysia", "lat": 3.1390, "lon": 101.6869, "tz": "Asia/Kuala_Lumpur"},
    "tokyo": {"city": "Tokyo", "country": "Japan", "lat": 35.6762, "lon": 139.6503, "tz": "Asia/Tokyo"},
    "london": {"city": "London", "country": "United Kingdom", "lat": 51.5074, "lon": -0.1278, "tz": "Europe/London"},
    "new york": {"city": "New York", "country": "United States", "lat": 40.7128, "lon": -74.0060, "tz": "America/New_York"},
    "singapore": {"city": "Singapore", "country": "Singapore", "lat": 1.3521, "lon": 103.8198, "tz": "Asia/Singapore"},
    "san francisco": {"city": "San Francisco", "country": "United States", "lat": 37.7749, "lon": -122.4194, "tz": "America/Los_Angeles"},
    "dubai": {"city": "Dubai", "country": "United Arab Emirates", "lat": 25.2048, "lon": 55.2708, "tz": "Asia/Dubai"},
    "sydney": {"city": "Sydney", "country": "Australia", "lat": -33.8688, "lon": 151.2093, "tz": "Australia/Sydney"},
    "berlin": {"city": "Berlin", "country": "Germany", "lat": 52.5200, "lon": 13.4050, "tz": "Europe/Berlin"},
}

# ---------------------------------------------------------------------------
# Pydantic Schemas (Preserved & Extended)
# ---------------------------------------------------------------------------
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
    classification: Optional[str] = None
    readiness_classification: Optional[str] = None
    focus_score: Optional[int] = None
    fatigue_score: Optional[int] = None
    decision_quality_score: Optional[int] = None

class AlertnessResponse(BaseModel):
    alertness_score: int
    score: int
    phase: str
    phase_name: str
    chronotype: Optional[str] = None
    effective_hour: Optional[float] = None

class DeepWorkResponse(BaseModel):
    deep_work_probability: float
    probability: float
    deep_work_score: Optional[int] = None
    status: Optional[str] = None

class ResponseProbabilityResponse(BaseModel):
    response_probability: float
    acceptance_probability: float
    dismissal_probability: float
    social_energy: Optional[int] = None

class SleepStateResponse(BaseModel):
    awake_probability: float
    sleep_probability: float
    sleep_debt_hours: Optional[float] = None
    sleep_debt_score: Optional[int] = None

class H3TensorResponse(BaseModel):
    h3: str
    alertness: int
    fatigue: int
    readiness: int
    sleep_debt: int
    status_code: str
    status: str
    classification: Optional[str] = None
    decision_quality: Optional[int] = None

class HumanOSSignalResponse(BaseModel):
    human_readiness: int
    alertness: int
    fatigue: int
    response_probability: float
    deep_work_probability: float
    social_energy: int
    status: str
    # Extended unified agent signals
    human_readiness_index: Optional[int] = None
    focus: Optional[int] = None
    decision_quality: Optional[int] = None
    deep_work_score: Optional[int] = None
    meeting_score: Optional[int] = None
    chronotype_adjustment: Optional[float] = None
    classification: Optional[str] = None
    readiness_classification: Optional[str] = None

class CompareRegionsInput(BaseModel):
    cities: List[str] = Field(
        default=["Kuala Lumpur", "Tokyo", "London", "New York"],
        description="List of city names to evaluate"
    )

# ---------------------------------------------------------------------------
# Cross-Version H3 Universal Wrappers
# ---------------------------------------------------------------------------
def _is_valid_h3(cell: str) -> bool:
    try:
        if hasattr(h3, 'is_valid_cell'):
            return bool(h3.is_valid_cell(cell))
        elif hasattr(h3, 'h3_is_valid'):
            return bool(h3.h3_is_valid(cell))
    except Exception:
        pass
    return len(cell) >= 15 and all(c in '0123456789abcdefABCDEF' for c in cell)

def _h3_to_latlon(cell: str) -> Tuple[float, float]:
    try:
        if hasattr(h3, 'cell_to_latlng'):
            coords = h3.cell_to_latlng(cell)
            return float(coords[0]), float(coords[1])
        elif hasattr(h3, 'h3_to_geo'):
            coords = h3.h3_to_geo(cell)
            return float(coords[0]), float(coords[1])
    except Exception:
        pass
    return 2.0442, 102.5689

# ---------------------------------------------------------------------------
# Core Analytical Biological State Engines
# ---------------------------------------------------------------------------
def calculate_chronotype_shift(chronotype: str = "NEUTRAL") -> float:
    """
    Step 2: Chronotype Engine.
    Computes circadian phase shift in hours:
    EARLY_BIRD (-2.0h), NEUTRAL (0.0h), NIGHT_OWL (+2.5h).
    """
    c = str(chronotype).strip().upper()
    if c in ["EARLY_BIRD", "LARK", "MORNING"]:
        return -2.0
    elif c in ["NIGHT_OWL", "OWL", "EVENING"]:
        return 2.5
    return 0.0

def calculate_alertness(
    local_time: Optional[float] = None,
    chronotype: str = "NEUTRAL",
    lat: Optional[float] = None,
    lon: Optional[float] = None,
    *args: Any,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Step 1: Alertness Engine.
    Smooth circadian Process C & Process S model combining diurnal activation sigmoids,
    morning peak (~10:30), postprandial dip (~14:00), and evening recovery (~17:30).
    """
    if local_time is None:
        if lat is not None and lon is not None:
            local_time = get_local_hour(lat, lon)
        else:
            local_time = 12.0

    shift = calculate_chronotype_shift(chronotype)
    eff_h = (local_time - shift) % 24.0

    # Smooth daytime activation curve
    day_activation = 1.0 / (1.0 + math.exp(-1.1 * (eff_h - 7.0))) * (1.0 / (1.0 + math.exp(0.9 * (eff_h - 22.0))))

    # Gaussian peak and dip components
    p1 = 28.0 * math.exp(-((eff_h - 10.5) ** 2) / 8.0)
    p2 = 22.0 * math.exp(-((eff_h - 17.5) ** 2) / 6.0)
    dip = 10.0 * math.exp(-((eff_h - 14.0) ** 2) / 3.0)
    night_floor = 10.0 + 8.0 * (1.0 - (1.0 / (1.0 + math.exp(-0.8 * (eff_h - 4.0)))))

    raw = (day_activation * 52.0) + p1 + p2 - dip + night_floor
    alertness_score = int(round(float(np.clip(raw, 10.0, 98.0))))

    if 9.0 <= eff_h < 13.5:
        phase = AlertnessPhase.PEAK_ALERTNESS.value
    elif 13.5 <= eff_h < 16.5:
        phase = AlertnessPhase.AFTERNOON_PLATEAU.value
    elif 16.5 <= eff_h < 20.5:
        phase = AlertnessPhase.EVENING_BOOST.value
    elif 20.5 <= eff_h < 23.0:
        phase = AlertnessPhase.WIND_DOWN.value
    elif 23.0 <= eff_h or eff_h < 6.5:
        phase = AlertnessPhase.SLEEP_PERIOD.value
    else:
        phase = AlertnessPhase.MORNING_RAMP.value

    return {
        "alertness_score": alertness_score,
        "score": alertness_score,
        "phase": phase,
        "phase_name": phase,
        "effective_hour": round(eff_h, 2),
        "chronotype": chronotype.upper(),
        "chronotype_shift_hours": shift,
    }

def calculate_sleep_debt(
    recommended_sleep: float = 8.0,
    actual_sleep: float = 7.0,
    *args: Any,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Step 4: Sleep Debt Model.
    Outputs: sleep_debt_hours and normalized sleep_debt_score (0-100).
    """
    rec = max(4.0, min(12.0, float(recommended_sleep)))
    act = max(0.0, min(14.0, float(actual_sleep)))
    debt_hours = max(0.0, rec - act)
    debt_score = float(np.clip((debt_hours / 4.0) * 100.0, 0.0, 100.0))

    if debt_score <= 15.0:
        status = "MINIMAL"
    elif debt_score <= 45.0:
        status = "MODERATE"
    elif debt_score <= 75.0:
        status = "HIGH"
    else:
        status = "CRITICAL"

    return {
        "recommended_sleep": round(rec, 1),
        "actual_sleep": round(act, 1),
        "sleep_debt_hours": round(debt_hours, 2),
        "sleep_debt_score": int(round(debt_score)),
        "status": status,
    }

def calculate_fatigue(
    time_awake: float = 6.0,
    sleep_debt: float = 1.0,
    local_time: Optional[float] = None,
    chronotype: str = "NEUTRAL",
    *args: Any,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Step 3: Fatigue Engine.
    Models homeostatic sleep pressure (Process S) and circadian fatigue modulation.
    """
    t_awake = max(0.0, float(time_awake))
    debt = max(0.0, float(sleep_debt))

    homeostatic = min(1.0, (t_awake / 16.0) ** 1.3)
    homeostatic_score = homeostatic * 70.0
    debt_score = min(30.0, debt * 10.0)

    if local_time is not None:
        alert_dict = calculate_alertness(local_time=local_time, chronotype=chronotype)
        circadian_fatigue = (100.0 - alert_dict["alertness_score"]) * 0.45
    else:
        circadian_fatigue = 20.0

    raw_fatigue = 0.45 * homeostatic_score + 0.30 * debt_score + 0.25 * circadian_fatigue
    fatigue_score = int(round(float(np.clip(raw_fatigue * 1.35, 5.0, 99.0))))

    if fatigue_score < 30:
        risk = "LOW"
    elif fatigue_score < 65:
        risk = "MODERATE"
    elif fatigue_score < 85:
        risk = "HIGH"
    else:
        risk = "CRITICAL"

    return {
        "fatigue_score": fatigue_score,
        "score": fatigue_score,
        "risk": risk,
        "fatigue_level": risk,
        "time_awake_hours": round(t_awake, 1),
        "sleep_debt_hours": round(debt, 1),
    }

def calculate_cognitive_capacity(
    alertness_score: float = 75.0,
    fatigue_score: float = 25.0,
    sleep_debt_score: float = 20.0,
    *args: Any,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Step 5: Cognitive Performance Engine.
    Outputs: focus_score, memory_score, and reasoning_score (Range: 0-100).
    """
    a = max(0.0, min(100.0, float(alertness_score)))
    f = max(0.0, min(100.0, float(fatigue_score)))
    d = max(0.0, min(100.0, float(sleep_debt_score)))
    inv_f = 100.0 - f
    inv_d = 100.0 - d

    focus = 0.65 * a + 0.35 * inv_f
    memory = 0.45 * a + 0.30 * inv_f + 0.25 * inv_d
    reasoning = 0.55 * a + 0.30 * inv_f + 0.15 * inv_d

    f_score = int(round(float(np.clip(focus, 5.0, 99.0))))
    m_score = int(round(float(np.clip(memory, 5.0, 99.0))))
    r_score = int(round(float(np.clip(reasoning, 5.0, 99.0))))
    composite = int(round((f_score + m_score + r_score) / 3.0))

    return {
        "focus_score": f_score,
        "memory_score": m_score,
        "reasoning_score": r_score,
        "cognitive_capacity_score": composite,
        "score": composite,
    }

def calculate_deep_work_potential(
    focus_score: float = 75.0,
    alertness: float = 75.0,
    fatigue: float = 25.0,
    *args: Any,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Step 6: Deep Work Engine.
    Predicts focus capacity and returns deep_work_score (0-100) and probability (0.0-1.0).
    """
    f = max(0.0, min(100.0, float(focus_score)))
    a = max(0.0, min(100.0, float(alertness)))
    fat = max(0.0, min(100.0, float(fatigue)))

    raw = 0.50 * f + 0.30 * a + 0.20 * (100.0 - fat)
    score = int(round(float(np.clip(raw, 5.0, 99.0))))
    prob = round(score / 100.0, 2)

    if score >= 80:
        status = "OPTIMAL_FLOW"
    elif score >= 60:
        status = "FAVORABLE"
    elif score >= 40:
        status = "MODERATE"
    else:
        status = "UNFAVORABLE"

    return {
        "deep_work_score": score,
        "deep_work_probability": prob,
        "probability": prob,
        "status": status,
    }

def calculate_meeting_readiness(
    alertness: float = 75.0,
    fatigue: float = 25.0,
    focus: float = 75.0,
    social_energy: float = 70.0,
    *args: Any,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Step 7: Meeting Readiness Engine.
    Evaluates emotional engagement, conversational alertness, and executive presence.
    """
    a = max(0.0, min(100.0, float(alertness)))
    fat = max(0.0, min(100.0, float(fatigue)))
    foc = max(0.0, min(100.0, float(focus)))
    soc = max(0.0, min(100.0, float(social_energy)))

    raw = 0.35 * a + 0.25 * (100.0 - fat) + 0.20 * foc + 0.20 * soc
    meeting_score = int(round(float(np.clip(raw, 5.0, 99.0))))

    if meeting_score >= 80:
        status = "IDEAL_FOR_MEETINGS"
    elif meeting_score >= 60:
        status = "SUITABLE"
    elif meeting_score >= 40:
        status = "ACCEPTABLE"
    else:
        status = "POOR"

    return {
        "meeting_score": meeting_score,
        "meeting_readiness_score": meeting_score,
        "status": status,
    }

def calculate_decision_quality(
    reasoning_score: float = 75.0,
    fatigue: float = 25.0,
    alertness: float = 75.0,
    sleep_debt: float = 20.0,
    *args: Any,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Step 8: Decision Quality Engine.
    Used by agents for strategy, purchasing, negotiation, and high-stakes risk evaluation.
    """
    r = max(0.0, min(100.0, float(reasoning_score)))
    fat = max(0.0, min(100.0, float(fatigue)))
    a = max(0.0, min(100.0, float(alertness)))
    d = max(0.0, min(100.0, float(sleep_debt)))

    raw = 0.45 * r + 0.25 * (100.0 - fat) + 0.15 * a + 0.15 * (100.0 - d)
    decision_quality_score = int(round(float(np.clip(raw, 5.0, 99.0))))

    if decision_quality_score >= 85:
        recommendation = "OPTIMAL_FOR_HIGH_STAKES_DECISIONS"
    elif decision_quality_score >= 70:
        recommendation = "SUITABLE_FOR_STRATEGIC_CHOICES"
    elif decision_quality_score >= 50:
        recommendation = "ROUTINE_DECISIONS_ONLY"
    else:
        recommendation = "DEFER_CRITICAL_DECISIONS"

    return {
        "decision_quality_score": decision_quality_score,
        "recommendation": recommendation,
        "status": recommendation,
    }

def determine_readiness_classification(readiness_score: float) -> str:
    """
    Step 10: Standard Readiness Classification.
    Returns: ASLEEP, RECOVERING, ACTIVE, HIGH_PERFORMANCE, PEAK_COGNITIVE_STATE.
    """
    score = float(readiness_score)
    if score >= 85.0:
        return ReadinessClassification.PEAK_COGNITIVE_STATE.value
    elif score >= 70.0:
        return ReadinessClassification.HIGH_PERFORMANCE.value
    elif score >= 50.0:
        return ReadinessClassification.ACTIVE.value
    elif score >= 30.0:
        return ReadinessClassification.RECOVERING.value
    else:
        return ReadinessClassification.ASLEEP.value

def calculate_human_readiness_index(
    alertness: float,
    focus: float,
    fatigue: float,
    sleep_debt: float,
    decision_quality: float,
    *args: Any,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Step 9: Human Readiness Index (HRI).
    Primary machine-consumable signal for autonomous agents (0-100).
    """
    a = max(0.0, min(100.0, float(alertness)))
    foc = max(0.0, min(100.0, float(focus)))
    fat = max(0.0, min(100.0, float(fatigue)))
    debt = max(0.0, min(100.0, float(sleep_debt)))
    dq = max(0.0, min(100.0, float(decision_quality)))

    raw = 0.25 * a + 0.25 * foc + 0.20 * (100.0 - fat) + 0.15 * (100.0 - debt) + 0.15 * dq
    readiness_index = int(round(float(np.clip(raw, 5.0, 99.0))))
    classification = determine_readiness_classification(readiness_index)
    legacy_state = "OPTIMAL" if readiness_index >= 80 else ("GOOD" if readiness_index >= 60 else ("FAIR" if readiness_index >= 40 else "LOW"))

    return {
        "human_readiness_index": readiness_index,
        "readiness": readiness_index,
        "human_readiness": readiness_index,
        "classification": classification,
        "readiness_classification": classification,
        "state": legacy_state,
        "status": legacy_state,
    }

# ---------------------------------------------------------------------------
# Preserved Internal Helper Functions (Backward-Compatible)
# ---------------------------------------------------------------------------
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

def compute_alertness_internal(lat: float, lon: float, chronotype: str = "NEUTRAL") -> Tuple[int, str]:
    hour = get_local_hour(lat, lon)
    res = calculate_alertness(local_time=hour, chronotype=chronotype)
    return res["alertness_score"], res["phase"]

def compute_fatigue_internal(lat: float, lon: float, chronotype: str = "NEUTRAL") -> Tuple[int, str]:
    hour = get_local_hour(lat, lon)
    # Estimate awake time from diurnal hour
    eff_awake = max(1.0, (hour - 6.5) % 24.0) if hour >= 6.5 else (hour + 17.5)
    fatigue_res = calculate_fatigue(time_awake=eff_awake, sleep_debt=1.0, local_time=hour, chronotype=chronotype)
    return fatigue_res["fatigue_score"], fatigue_res["risk"]

def compute_deep_work_internal(lat: float, lon: float, chronotype: str = "NEUTRAL") -> float:
    hour = get_local_hour(lat, lon)
    alert_dict = calculate_alertness(local_time=hour, chronotype=chronotype)
    fatigue_score, _ = compute_fatigue_internal(lat, lon, chronotype)
    cog_res = calculate_cognitive_capacity(alertness_score=alert_dict["alertness_score"], fatigue_score=fatigue_score)
    deep_res = calculate_deep_work_potential(
        focus_score=cog_res["focus_score"],
        alertness=alert_dict["alertness_score"],
        fatigue=fatigue_score,
    )
    return deep_res["deep_work_probability"]

def compute_response_probability_internal(lat: float, lon: float, chronotype: str = "NEUTRAL") -> Tuple[float, float, float]:
    alertness, _ = compute_alertness_internal(lat, lon, chronotype)
    fatigue, _ = compute_fatigue_internal(lat, lon, chronotype)
    acceptance = round(float(np.clip((0.65 * alertness + 0.35 * (100.0 - fatigue)) / 100.0, 0.05, 0.95)), 2)
    dismissal = round(1.0 - acceptance, 2)
    return acceptance, acceptance, dismissal

def compute_sleep_state_internal(lat: float, lon: float, chronotype: str = "NEUTRAL") -> Tuple[float, float]:
    hour = get_local_hour(lat, lon)
    shift = calculate_chronotype_shift(chronotype)
    eff_h = (hour - shift) % 24.0

    if 23.0 <= eff_h or eff_h < 6.0:
        sleep_prob = 0.92
    elif 6.0 <= eff_h < 7.5:
        sleep_prob = 0.40
    elif 21.5 <= eff_h < 23.0:
        sleep_prob = 0.35
    else:
        sleep_prob = 0.06

    awake_prob = round(1.0 - sleep_prob, 2)
    return awake_prob, sleep_prob

def compute_social_energy_internal(lat: float, lon: float, chronotype: str = "NEUTRAL") -> int:
    hour = get_local_hour(lat, lon)
    shift = calculate_chronotype_shift(chronotype)
    eff_h = (hour - shift) % 24.0

    if 10.0 <= eff_h <= 20.0:
        return 82
    elif (8.0 <= eff_h < 10.0) or (20.0 < eff_h <= 22.5):
        return 65
    else:
        return 22

def compute_cognitive_capacity_internal(lat: float, lon: float, chronotype: str = "NEUTRAL") -> int:
    alertness, _ = compute_alertness_internal(lat, lon, chronotype)
    fatigue, _ = compute_fatigue_internal(lat, lon, chronotype)
    cog = calculate_cognitive_capacity(alertness_score=alertness, fatigue_score=fatigue)
    return cog["cognitive_capacity_score"]

def compute_sleep_debt_internal(lat: float, lon: float) -> int:
    debt_res = calculate_sleep_debt(recommended_sleep=8.0, actual_sleep=6.8)
    return debt_res["sleep_debt_score"]

def compute_human_readiness_internal(lat: float, lon: float, chronotype: str = "NEUTRAL") -> Tuple[int, str]:
    alertness, _ = compute_alertness_internal(lat, lon, chronotype)
    fatigue, _ = compute_fatigue_internal(lat, lon, chronotype)
    debt_score = compute_sleep_debt_internal(lat, lon)
    cog = calculate_cognitive_capacity(alertness_score=alertness, fatigue_score=fatigue, sleep_debt_score=debt_score)
    dq = calculate_decision_quality(reasoning_score=cog["reasoning_score"], fatigue=fatigue, alertness=alertness, sleep_debt=debt_score)

    hri_res = calculate_human_readiness_index(
        alertness=alertness,
        focus=cog["focus_score"],
        fatigue=fatigue,
        sleep_debt=debt_score,
        decision_quality=dq["decision_quality_score"],
    )
    return hri_res["human_readiness_index"], hri_res["state"]

# ---------------------------------------------------------------------------
# Step 11: Multi-Timezone Workforce Intelligence Engine
# ---------------------------------------------------------------------------
def compare_regions(cities: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Step 11: Compares real-time workforce readiness across global business centers.
    Returns best regions for meetings, sales, planning, and deep work.
    """
    if not cities:
        cities = ["Kuala Lumpur", "Tokyo", "London", "New York"]

    evaluated_regions = []

    for name in cities:
        clean = name.strip().lower()
        if clean in GLOBAL_WORKFORCE_REGISTRY:
            meta = GLOBAL_WORKFORCE_REGISTRY[clean]
        else:
            matched = False
            for k, val in GLOBAL_WORKFORCE_REGISTRY.items():
                if clean in k or k in clean:
                    meta = val
                    matched = True
                    break
            if not matched:
                meta = {"city": name.strip().title(), "country": "Global", "lat": 3.1390, "lon": 101.6869, "tz": "UTC"}

        lat, lon = meta["lat"], meta["lon"]
        hour = get_local_hour(lat, lon)
        alert_dict = calculate_alertness(local_time=hour)
        fatigue_res = calculate_fatigue(time_awake=max(1.0, (hour - 6.5) % 24.0), local_time=hour)
        debt_res = calculate_sleep_debt(8.0, 7.0)
        cog = calculate_cognitive_capacity(alert_dict["alertness_score"], fatigue_res["fatigue_score"], debt_res["sleep_debt_score"])
        deep_res = calculate_deep_work_potential(cog["focus_score"], alert_dict["alertness_score"], fatigue_res["fatigue_score"])
        meeting_res = calculate_meeting_readiness(alert_dict["alertness_score"], fatigue_res["fatigue_score"], cog["focus_score"], 80.0)
        dq_res = calculate_decision_quality(cog["reasoning_score"], fatigue_res["fatigue_score"], alert_dict["alertness_score"], debt_res["sleep_debt_score"])

        hri = calculate_human_readiness_index(
            alertness=alert_dict["alertness_score"],
            focus=cog["focus_score"],
            fatigue=fatigue_res["fatigue_score"],
            sleep_debt=debt_res["sleep_debt_score"],
            decision_quality=dq_res["decision_quality_score"],
        )

        evaluated_regions.append({
            "city": meta["city"],
            "country": meta["country"],
            "timezone": meta.get("tz", "UTC"),
            "local_time_hour": round(hour, 2),
            "human_readiness_index": hri["human_readiness_index"],
            "alertness_score": alert_dict["alertness_score"],
            "focus_score": cog["focus_score"],
            "fatigue_score": fatigue_res["fatigue_score"],
            "deep_work_score": deep_res["deep_work_score"],
            "meeting_score": meeting_res["meeting_score"],
            "decision_quality_score": dq_res["decision_quality_score"],
            "readiness_classification": hri["classification"],
        })

    best_for_meetings = max(evaluated_regions, key=lambda x: x["meeting_score"])
    best_for_sales = max(evaluated_regions, key=lambda x: (x["alertness_score"] + x["meeting_score"]))
    best_for_planning = max(evaluated_regions, key=lambda x: x["decision_quality_score"])
    best_for_deep_work = max(evaluated_regions, key=lambda x: x["deep_work_score"])

    return {
        "best_for_meetings": best_for_meetings["city"],
        "best_for_sales": best_for_sales["city"],
        "best_for_planning": best_for_planning["city"],
        "best_for_deep_work": best_for_deep_work["city"],
        "regions": evaluated_regions,
    }

# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------
@app.get("/")
def read_root() -> Dict[str, Any]:
    return {
        "status": "healthy",
        "service": "circadian-os-3882",
        "version": "2.0.0",
        "mission": "Human Biological State & Cognitive Readiness Intelligence Grid",
    }

@app.get("/health")
def health_check() -> Dict[str, Any]:
    return {"status": "healthy"}

@app.get("/tools")
def list_tools() -> Dict[str, Any]:
    """MCP & Tool discovery catalog."""
    return {
        "tools": [
            {
                "name": "get_human_readiness",
                "description": "Returns biological readiness score, cognitive capacity, and readiness classification.",
                "input_schema": {"lat": "float", "lon": "float", "chronotype": "string"},
            },
            {
                "name": "get_alertness",
                "description": "Returns mathematical circadian alertness state and phase.",
                "input_schema": {"lat": "float", "lon": "float", "chronotype": "string"},
            },
            {
                "name": "get_deep_work_window",
                "description": "Predicts deep work potential and concentration probability.",
                "input_schema": {"lat": "float", "lon": "float"},
            },
            {
                "name": "get_response_probability",
                "description": "Predicts human response and acceptance likelihood.",
                "input_schema": {"lat": "float", "lon": "float"},
            },
            {
                "name": "get_sleep_state",
                "description": "Returns sleep and wakefulness estimate.",
                "input_schema": {"lat": "float", "lon": "float"},
            },
            {
                "name": "get_h3_circadian_tensor",
                "description": "Returns circadian intelligence tensor for H3 discrete hexagon.",
                "input_schema": {"h3": "string"},
            },
            {
                "name": "get_humanos_signal",
                "description": "Unified HumanOS operational signal for autonomous scheduling and business agents.",
                "input_schema": {"lat": "float", "lon": "float", "chronotype": "string"},
            },
            {
                "name": "compare_regions",
                "description": "Compares workforce biological readiness across global metropolitan centers.",
                "input_schema": {"cities": "list[string]"},
            },
        ]
    }

@app.get("/tools/get_human_readiness", response_model=HumanReadinessResponse)
def get_human_readiness(
    lat: float = Query(...),
    lon: float = Query(...),
    chronotype: str = Query("NEUTRAL"),
):
    validate_coordinates(lat, lon)
    score, state = compute_human_readiness_internal(lat, lon, chronotype)
    classification = determine_readiness_classification(score)
    alertness, _ = compute_alertness_internal(lat, lon, chronotype)
    fatigue, _ = compute_fatigue_internal(lat, lon, chronotype)
    cog = calculate_cognitive_capacity(alertness_score=alertness, fatigue_score=fatigue)
    dq = calculate_decision_quality(cog["reasoning_score"], fatigue, alertness, 25.0)

    return {
        "human_readiness": score,
        "readiness": score,
        "state": state,
        "status": state,
        "classification": classification,
        "readiness_classification": classification,
        "focus_score": cog["focus_score"],
        "fatigue_score": fatigue,
        "decision_quality_score": dq["decision_quality_score"],
    }

@app.get("/tools/get_alertness", response_model=AlertnessResponse)
def get_alertness(
    lat: float = Query(...),
    lon: float = Query(...),
    chronotype: str = Query("NEUTRAL"),
):
    validate_coordinates(lat, lon)
    score, phase = compute_alertness_internal(lat, lon, chronotype)
    hour = get_local_hour(lat, lon)
    shift = calculate_chronotype_shift(chronotype)

    return {
        "alertness_score": score,
        "score": score,
        "phase": phase,
        "phase_name": phase,
        "chronotype": chronotype.upper(),
        "effective_hour": round((hour - shift) % 24.0, 2),
    }

@app.get("/tools/get_deep_work_window", response_model=DeepWorkResponse)
def get_deep_work_window(lat: float = Query(...), lon: float = Query(...)):
    validate_coordinates(lat, lon)
    prob = compute_deep_work_internal(lat, lon)
    score = int(round(prob * 100))
    status = "OPTIMAL_FLOW" if score >= 80 else ("FAVORABLE" if score >= 60 else "MODERATE")

    return {
        "deep_work_probability": prob,
        "probability": prob,
        "deep_work_score": score,
        "status": status,
    }

@app.get("/tools/get_response_probability", response_model=ResponseProbabilityResponse)
def get_response_probability(lat: float = Query(...), lon: float = Query(...)):
    validate_coordinates(lat, lon)
    resp_prob, acceptance, dismissal = compute_response_probability_internal(lat, lon)
    social_energy = compute_social_energy_internal(lat, lon)

    return {
        "response_probability": resp_prob,
        "acceptance_probability": acceptance,
        "dismissal_probability": dismissal,
        "social_energy": social_energy,
    }

@app.get("/tools/get_sleep_state", response_model=SleepStateResponse)
def get_sleep_state(lat: float = Query(...), lon: float = Query(...)):
    validate_coordinates(lat, lon)
    awake_prob, sleep_prob = compute_sleep_state_internal(lat, lon)
    debt_res = calculate_sleep_debt(8.0, 7.0)

    return {
        "awake_probability": awake_prob,
        "sleep_probability": sleep_prob,
        "sleep_debt_hours": debt_res["sleep_debt_hours"],
        "sleep_debt_score": debt_res["sleep_debt_score"],
    }

@app.get("/tools/get_h3_circadian_tensor", response_model=H3TensorResponse)
def get_h3_circadian_tensor(h3_index: str = Query(..., alias="h3")):
    if not _is_valid_h3(h3_index):
        lat, lon = 2.0442, 102.5689
    else:
        lat, lon = _h3_to_latlon(h3_index)

    alertness, _ = compute_alertness_internal(lat, lon)
    fatigue, _ = compute_fatigue_internal(lat, lon)
    readiness, _ = compute_human_readiness_internal(lat, lon)
    sleep_debt = compute_sleep_debt_internal(lat, lon)
    classification = determine_readiness_classification(readiness)
    cog = calculate_cognitive_capacity(alertness_score=alertness, fatigue_score=fatigue, sleep_debt_score=sleep_debt)
    dq = calculate_decision_quality(cog["reasoning_score"], fatigue, alertness, sleep_debt)
    deep_prob = compute_deep_work_internal(lat, lon)

    return {
        "h3": h3_index,
        "alertness": alertness,
        "fatigue": fatigue,
        "readiness": readiness,
        "sleep_debt": sleep_debt,
        "status_code": "OK",
        "status": "ACTIVE",
        "classification": classification,
        "decision_quality": dq["decision_quality_score"],
        "deep_work_probability": deep_prob,
    }

@app.get("/tools/get_humanos_signal", response_model=HumanOSSignalResponse)
def get_humanos_signal(
    lat: float = Query(...),
    lon: float = Query(...),
    chronotype: str = Query("NEUTRAL"),
):
    """
    Step 12: Unified HumanOS Machine Signal.
    Primary machine-consumable interface exposing biological readiness, focus,
    decision quality, and meeting feasibility.
    """
    validate_coordinates(lat, lon)
    readiness, legacy_state = compute_human_readiness_internal(lat, lon, chronotype)
    alertness, _ = compute_alertness_internal(lat, lon, chronotype)
    fatigue, _ = compute_fatigue_internal(lat, lon, chronotype)
    resp_prob, _, _ = compute_response_probability_internal(lat, lon, chronotype)
    deep_work = compute_deep_work_internal(lat, lon, chronotype)
    social_energy = compute_social_energy_internal(lat, lon, chronotype)

    # Calculate rich cognitive signals
    debt_score = compute_sleep_debt_internal(lat, lon)
    cog = calculate_cognitive_capacity(alertness_score=alertness, fatigue_score=fatigue, sleep_debt_score=debt_score)
    deep_pot = calculate_deep_work_potential(cog["focus_score"], alertness, fatigue)
    meeting_res = calculate_meeting_readiness(alertness, fatigue, cog["focus_score"], social_energy)
    dq_res = calculate_decision_quality(cog["reasoning_score"], fatigue, alertness, debt_score)
    classification = determine_readiness_classification(readiness)
    shift = calculate_chronotype_shift(chronotype)

    return {
        "human_readiness": readiness,
        "alertness": alertness,
        "fatigue": fatigue,
        "response_probability": float(round(resp_prob * 100.0, 1)),
        "deep_work_probability": float(round(deep_work * 100.0, 1)),
        "social_energy": social_energy,
        "status": legacy_state,
        "human_readiness_index": readiness,
        "focus": cog["focus_score"],
        "decision_quality": dq_res["decision_quality_score"],
        "deep_work_score": deep_pot["deep_work_score"],
        "meeting_score": meeting_res["meeting_score"],
        "chronotype_adjustment": shift,
        "classification": classification,
        "readiness_classification": classification,
    }

@app.get("/tools/compare_regions")
def api_compare_regions(
    cities: List[str] = Query(default=["Kuala Lumpur", "Tokyo", "London", "New York"])
) -> Dict[str, Any]:
    """Workforce biological readiness comparison endpoint across timezones."""
    return compare_regions(cities=cities)

@app.post("/tools/compare_regions")
def api_compare_regions_post(payload: CompareRegionsInput) -> Dict[str, Any]:
    return compare_regions(cities=payload.cities)

# ---------------------------------------------------------------------------
# CLI & Server Runner
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
