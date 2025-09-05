# tests/test_unep_api.py
import os, json, pytest, requests
from dotenv import load_dotenv

load_dotenv()

BASE_URL = os.getenv("UNEP_BASE_URL", "https://methanedata.unep.org").rstrip("/")
API_KEY = os.getenv("UNEP_API_KEY")
ENDPOINT_PATH = os.getenv("UNEP_ENDPOINT_PATH", "/api/countries")  # seen in /api/docs
RUN_API_TESTS = os.getenv("RUN_API_TESTS", "1") == "1"

def _headers():
    return {
        "Authorization": f"Bearer {API_KEY}",
        "Accept": "application/json",
        "User-Agent": "FieldMetrics/1.0 (+https://fieldmetrics.io)",
        "Referer": "https://methanedata.unep.org/api/docs",
        "Origin": "https://methanedata.unep.org",
    }

def test_env_has_unep_api_key():
    assert API_KEY and API_KEY.strip(), "UNEP_API_KEY is missing"
    assert len(API_KEY) >= 8

@pytest.mark.skipif(not RUN_API_TESTS, reason="Set RUN_API_TESTS=1 to run live API smoke test")
def test_eye_on_methane_api_smoke():
    url = f"{BASE_URL}{ENDPOINT_PATH}"
    resp = requests.get(url, headers=_headers(), timeout=25)

    head = resp.text[:400].replace("\n", " ")
    assert resp.status_code == 200, f"Unexpected {resp.status_code} for {url} | Body: {head}"

    try:
        data = resp.json()
    except json.JSONDecodeError as e:
        pytest.fail(f"Non-JSON response: {e} | Body: {head}")

    assert data is not None
    # very light sanity
    assert isinstance(data, (list, dict))