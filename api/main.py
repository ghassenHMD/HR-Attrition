"""API d'interprétation – Attrition des employés (IBM HR)."""
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import shap
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from lime.lime_tabular import LimeTabularExplainer
from pydantic import BaseModel, Field
from sklearn.inspection import permutation_importance

# ------------------------------------------------------------------ chargement
B = joblib.load(Path(__file__).resolve().parent.parent / "artifacts" / "bundle.joblib")
MODEL, PRE, NAME = B["model"], B["preprocess"], B["model_name"]
FEATS, THRESHOLD = B["feat_names"], B["threshold"]
META = {f["name"]: f for f in B["features_meta"]}
RAW_COLS = list(META)
CAT = [c for c in META if META[c]["type"] == "categorical"]
X_TRAIN, X_TEST, Y_TEST, X_TEST_RAW = B["X_train_t"], B["X_test_t"], B["y_test"], B["X_test_raw"]

if NAME == "Logistic Regression":
    EXPLAINER = shap.LinearExplainer(MODEL, X_TRAIN)
else:
    EXPLAINER = shap.TreeExplainer(MODEL)
UNIT = "probabilité" if NAME in ("Random Forest", "Decision Tree") else "log-odds"

LIME = LimeTabularExplainer(X_TRAIN, feature_names=FEATS, class_names=["Reste", "Part"],
                            mode="classification", discretize_continuous=True, random_state=42)

# ------------------------------------------------------------------ schémas
class EmployeeRequest(BaseModel):
    features: dict[str, Any] = Field(
        ..., description="Caractéristiques de l'employé. Les champs absents prennent la valeur médiane/modale.")
    threshold: float | None = Field(None, ge=0, le=1, description="Seuil de décision (défaut : seuil du modèle).")
    top_k: int = Field(8, ge=1, le=30, description="Nombre de variables à retourner dans l'explication.")

    model_config = {"json_schema_extra": {"example": {
        "features": {"Age": 29, "OverTime": "Yes", "MonthlyIncome": 2800, "JobSatisfaction": 1,
                     "BusinessTravel": "Travel_Frequently", "MaritalStatus": "Single"},
        "threshold": 0.5, "top_k": 8}}}


class Contribution(BaseModel):
    feature: str
    label: str
    contribution: float
    direction: str


class PredictResponse(BaseModel):
    probability: float
    threshold: float
    decision: int
    decision_label: str
    risk_level: str
    model: str
    ignored_fields: list[str] = []


class ShapResponse(PredictResponse):
    base_value: float
    unit: str
    contributions: list[Contribution]
    summary: str


class LimeResponse(PredictResponse):
    contributions: list[Contribution]


class CompareResponse(PredictResponse):
    shap: list[Contribution]
    lime: list[Contribution]
    common_features: list[str]
    overlap_ratio: float
    same_sign_ratio: float
    summary: str


# ------------------------------------------------------------------ utilitaires
def to_frame(features: dict) -> tuple[pd.DataFrame, list[str]]:
    """Complète, valide et met en forme l'individu."""
    ignored = [k for k in features if k not in META]
    row = {c: META[c]["default"] for c in RAW_COLS}
    row.update({k: v for k, v in features.items() if k in META})
    for c, v in row.items():
        m = META[c]
        if m["type"] == "categorical" and v not in m["options"]:
            raise HTTPException(422, f"Valeur invalide pour '{c}' : {v!r}. Valeurs possibles : {m['options']}")
        if m["type"] == "numeric":
            try:
                row[c] = float(v)
            except (TypeError, ValueError):
                raise HTTPException(422, f"'{c}' doit être numérique (reçu : {v!r}).")
    return pd.DataFrame([row], columns=RAW_COLS), ignored


def prepare(req: EmployeeRequest):
    df, ignored = to_frame(req.features)
    x_df = pd.DataFrame(PRE.transform(df), columns=FEATS)
    p = float(MODEL.predict_proba(x_df)[0, 1])
    thr = THRESHOLD if req.threshold is None else req.threshold
    level = "élevé" if p >= thr else "modéré" if p >= thr / 2 else "faible"
    base = dict(probability=round(p, 4), threshold=round(thr, 3), decision=int(p >= thr),
                decision_label="Départ probable" if p >= thr else "Reste probable",
                risk_level=level, model=NAME, ignored_fields=ignored)
    return base, x_df, df.iloc[0].to_dict()


def pretty(feat: str, xv: float, raw: dict) -> str:
    if feat in raw:
        v = raw[feat]
        return f"{feat} = {v:g}" if isinstance(v, (int, float)) else f"{feat} = {v}"
    for c in CAT:
        if feat.startswith(c + "_"):
            mod = feat[len(c) + 1:]
            return f"{c} = {mod}" if xv > 0.5 else f"{c} ≠ {mod}"
    return feat


def direction(v: float) -> str:
    return "augmente le risque" if v > 0 else "diminue le risque"


def shap_positive(sv) -> np.ndarray:
    """Valeurs SHAP de la classe positive, quelle que soit la version de shap."""
    if isinstance(sv, list):
        return np.asarray(sv[1])
    sv = np.asarray(sv)
    return sv[:, :, 1] if sv.ndim == 3 else sv


def shap_contribs(x_df, raw, top_k):
    sv = shap_positive(EXPLAINER.shap_values(x_df))[0]
    base = float(np.atleast_1d(EXPLAINER.expected_value)[-1])
    order = np.argsort(-np.abs(sv))[:top_k]
    items = [Contribution(feature=FEATS[i], label=pretty(FEATS[i], x_df.iloc[0, i], raw),
                          contribution=round(float(sv[i]), 4), direction=direction(sv[i])) for i in order]
    return base, items


def lime_contribs(x_df, raw, top_k):
    fn = lambda a: MODEL.predict_proba(pd.DataFrame(a, columns=FEATS))
    exp = LIME.explain_instance(x_df.values[0], fn, num_features=top_k, num_samples=2000)
    rules = [r for r, _ in exp.as_list(label=1)]
    return [Contribution(feature=FEATS[i], label=rule, contribution=round(float(w), 4),
                         direction=direction(w)) for (i, w), rule in zip(exp.local_exp[1], rules)]


def sentence(base: dict, items: list[Contribution]) -> str:
    pos = [c.label for c in items if c.contribution > 0][:2]
    neg = [c.label for c in items if c.contribution < 0][:1]
    s = f"Le risque de départ est {base['risk_level']} ({base['probability']:.0%})"
    s += f", principalement à cause de {' et de '.join(pos)}." if pos else ". Aucun facteur n'augmente nettement ce risque."
    if neg:
        s += f" Il est atténué par {neg[0]}."
    return s


@lru_cache(maxsize=1)
def global_importance() -> list[dict]:
    X = pd.DataFrame(X_TEST, columns=FEATS)
    shap_abs = np.abs(shap_positive(EXPLAINER.shap_values(X))).mean(axis=0)
    perm = permutation_importance(MODEL, X, Y_TEST, scoring="roc_auc", n_repeats=5, random_state=42, n_jobs=1)
    rows = [{"feature": f, "shap_mean_abs": round(float(s), 4), "permutation": round(float(p), 4)}
            for f, s, p in zip(FEATS, shap_abs, perm.importances_mean)]
    return sorted(rows, key=lambda r: -r["shap_mean_abs"])


# ------------------------------------------------------------------ application
app = FastAPI(
    title="API d'interprétation – Attrition des employés",
    description=("Prédit le risque de départ d'un employé (IBM HR Analytics) et **explique** la prédiction "
                 "avec SHAP et LIME. Projet Data Mining – ENSI 2026-2027."),
    version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/", tags=["Système"], summary="État de l'API")
def health():
    return {"status": "ok", "model": NAME, "docs": "/docs"}


@app.get("/model/info", tags=["Modèle"], summary="Informations sur le modèle déployé")
def model_info():
    return {"model_name": NAME, "threshold": round(THRESHOLD, 3), "metrics": B["metrics"],
            "n_features_raw": len(RAW_COLS), "n_features_encoded": len(FEATS),
            "n_test_samples": len(X_TEST_RAW), "shap_unit": UNIT}


@app.get("/features", tags=["Modèle"], summary="Variables attendues (type, bornes, modalités, défaut)")
def features():
    return B["features_meta"]


@app.get("/samples/{idx}", tags=["Modèle"], summary="Un individu du jeu de test et son vrai label")
def sample(idx: int):
    if not 0 <= idx < len(X_TEST_RAW):
        raise HTTPException(404, f"Index hors limites (0 à {len(X_TEST_RAW) - 1}).")
    feats = {k: (v.item() if hasattr(v, "item") else v) for k, v in X_TEST_RAW.iloc[idx].to_dict().items()}
    return {"index": idx, "features": feats, "actual": int(Y_TEST[idx])}


@app.post("/predict", response_model=PredictResponse, tags=["Prédiction"],
          summary="Probabilité de départ et décision au seuil choisi")
def predict(req: EmployeeRequest):
    base, _, _ = prepare(req)
    return base


@app.post("/explain/shap", response_model=ShapResponse, tags=["Explications locales"],
          summary="Explication SHAP d'un individu (waterfall + phrase de synthèse)")
def explain_shap(req: EmployeeRequest):
    base, x_df, raw = prepare(req)
    bv, items = shap_contribs(x_df, raw, req.top_k)
    return {**base, "base_value": round(bv, 4), "unit": UNIT, "contributions": items,
            "summary": sentence(base, items)}


@app.post("/explain/lime", response_model=LimeResponse, tags=["Explications locales"],
          summary="Explication LIME d'un individu")
def explain_lime(req: EmployeeRequest):
    base, x_df, raw = prepare(req)
    return {**base, "contributions": lime_contribs(x_df, raw, req.top_k)}


@app.post("/explain/compare", response_model=CompareResponse, tags=["Explications locales"],
          summary="SHAP et LIME côte à côte, avec mesure d'accord")
def explain_compare(req: EmployeeRequest):
    base, x_df, raw = prepare(req)
    _, s_items = shap_contribs(x_df, raw, req.top_k)
    l_items = lime_contribs(x_df, raw, req.top_k)
    s_map = {c.feature: c.contribution for c in s_items}
    l_map = {c.feature: c.contribution for c in l_items}
    common = sorted(set(s_map) & set(l_map))
    same = [f for f in common if np.sign(s_map[f]) == np.sign(l_map[f])]
    return {**base, "shap": s_items, "lime": l_items, "common_features": common,
            "overlap_ratio": round(len(common) / req.top_k, 2),
            "same_sign_ratio": round(len(same) / len(common), 2) if common else 0.0,
            "summary": sentence(base, s_items)}


@app.get("/explain/global", tags=["Explications globales"],
         summary="Importance globale des variables (SHAP moyen |φ| et permutation)")
def explain_global(top_k: int = Query(15, ge=1, le=100)):
    return {"model": NAME, "importance": global_importance()[:top_k]}