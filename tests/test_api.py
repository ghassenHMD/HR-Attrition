from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)
EMP = {"features": {"OverTime": "Yes", "MonthlyIncome": 2500, "JobSatisfaction": 1, "MaritalStatus": "Single"}, "top_k": 6}


def test_health():
    assert client.get("/").json()["status"] == "ok"


def test_model_info_and_features():
    assert client.get("/model/info").json()["n_test_samples"] > 0
    f = client.get("/features").json()
    assert any(x["name"] == "OverTime" and x["label"] == "Heures supplémentaires" for x in f)


def test_predict_and_validation():
    r = client.post("/predict", json=EMP)
    assert r.status_code == 200 and 0 <= r.json()["probability"] <= 1
    assert client.post("/predict", json={"features": {"OverTime": "Peut-être"}}).status_code == 422


def test_shap_waterfall_is_additive():
    r = client.post("/explain/shap", json=EMP).json()
    assert len(r["contributions"]) == 6
    total = r["base_value"] + sum(c["contribution"] for c in r["contributions"]) + r["other_contribution"]
    assert abs(total - r["output_value"]) < 1e-3
    assert r["summary"].startswith("Le risque de départ")


def test_lime_and_compare():
    assert len(client.post("/explain/lime", json=EMP).json()["contributions"]) == 6
    c = client.post("/explain/compare", json=EMP).json()
    assert 0 <= c["overlap_ratio"] <= 1 and 0 <= c["same_sign_ratio"] <= 1


def test_global():
    g = client.get("/explain/global?top_k=10").json()
    assert len(g["shap"]) == 10 and len(g["permutation"]) >= 4
    sv = client.get("/explain/shap-values?top_k=5").json()
    assert len(sv["variables"]) == 5 and len(sv["variables"][0]["shap"]) == sv["n"]
    assert client.get("/explain/agreement?n=3").json()["n"] == 3


def test_samples_and_models():
    r = client.get("/samples/ranking?limit=5").json()
    assert len(r) == 5 and r[0]["probability"] >= r[-1]["probability"]
    assert client.get(f"/samples/{r[0]['index']}").status_code == 200
    ev = client.get("/models/evaluation").json()
    assert len(ev["proba"]) == 5
    t = client.post("/models/train", json={"model": "Decision Tree", "params": {"max_depth": 4}})
    assert t.status_code == 200 and len(t.json()["proba"]) == len(ev["y_test"])