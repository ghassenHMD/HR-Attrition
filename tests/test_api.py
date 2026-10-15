from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_health():
    assert client.get("/").json()["status"] == "ok"

def test_predict():
    r = client.post("/predict", json={"features": {"OverTime": "Yes"}})
    assert r.status_code == 200
    assert 0 <= r.json()["probability"] <= 1

def test_invalid_value():
    r = client.post("/predict", json={"features": {"OverTime": "Peut-être"}})
    assert r.status_code == 422

def test_shap():
    r = client.post("/explain/shap", json={"features": {"OverTime": "Yes"}, "top_k": 5})
    assert r.status_code == 200 and len(r.json()["contributions"]) == 5

def test_global():
    assert len(client.get("/explain/global?top_k=5").json()["importance"]) == 5