from fastapi.testclient import TestClient
from main import (
    app,
    compute_alertness_internal,
    compute_fatigue_internal,
    compute_deep_work_internal,
    compute_human_readiness_internal,
    compute_sleep_state_internal,
)

client = TestClient(app)

def test_root():
    assert client.get("/").status_code == 200

def test_health():
    assert client.get("/health").status_code == 200

def test_tools():
    assert client.get("/tools").status_code == 200

def test_endpoints():
    lat, lon = 2.0442, 102.5689
    assert client.get(f"/tools/get_human_readiness?lat={lat}&lon={lon}").status_code == 200
    assert client.get(f"/tools/get_alertness?lat={lat}&lon={lon}").status_code == 200
    assert client.get(f"/tools/get_deep_work_window?lat={lat}&lon={lon}").status_code == 200
    assert client.get(f"/tools/get_response_probability?lat={lat}&lon={lon}").status_code == 200

def test_offline_calculations():
    lat, lon = 2.0442, 102.5689
    score, phase = compute_alertness_internal(lat, lon)
    assert 0 <= score <= 100
    assert isinstance(phase, str)

    fatigue, risk = compute_fatigue_internal(lat, lon)
    assert 0 <= fatigue <= 100
    assert isinstance(risk, str)

    dw = compute_deep_work_internal(lat, lon)
    assert 0.0 <= dw <= 1.0

    readiness, state = compute_human_readiness_internal(lat, lon)
    assert 0 <= readiness <= 100
    assert isinstance(state, str)

    awake, sleep = compute_sleep_state_internal(lat, lon)
    assert 0.0 <= awake <= 1.0
    assert 0.0 <= sleep <= 1.0
