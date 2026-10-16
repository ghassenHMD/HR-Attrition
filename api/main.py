"""API d'interprétation – Attrition des employés (IBM HR Analytics)."""
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import joblib
import numpy as np
import pandas as pd
import shap
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from lime.lime_tabular import LimeTabularExplainer
from pydantic import BaseModel, Field
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

# ======================================================================= chargement
B = joblib.load(Path(__file__).resolve().parent.parent / "artifacts" / "bundle.joblib")
MODEL, PRE, NAME = B["model"], B["preprocess"], B["model_name"]
FEATS, THRESHOLD = B["feat_names"], B["threshold"]
META = {f["name"]: f for f in B["features_meta"]}
RAW_COLS = list(META)
CAT = [c for c in META if META[c]["type"] == "categorical"]
X_TRAIN, Y_TRAIN = B["X_train_t"], B["y_train"]
X_TEST, Y_TEST, X_TEST_RAW = B["X_test_t"], B["y_test"], B["X_test_raw"]
X_TEST_DF = pd.DataFrame(X_TEST, columns=FEATS)
PROBA_TEST = MODEL.predict_proba(X_TEST_DF)[:, 1]
SPW = float((Y_TRAIN == 0).sum() / (Y_TRAIN == 1).sum())

# Variables sensibles (équité) – l'origine n'existe pas dans ce jeu de données
SENSITIVE = [c for c in ["Age", "Gender", "MaritalStatus"] if c in META]

# Libellés lisibles pour un décideur RH
FR = {
    "Age": "Âge", "BusinessTravel": "Déplacements professionnels", "DailyRate": "Taux journalier",
    "Department": "Département", "DistanceFromHome": "Distance domicile-travail",
    "Education": "Niveau d'études", "EducationField": "Domaine d'études",
    "EnvironmentSatisfaction": "Satisfaction (environnement)", "Gender": "Genre",
    "HourlyRate": "Taux horaire", "JobInvolvement": "Implication", "JobLevel": "Niveau hiérarchique",
    "JobRole": "Poste", "JobSatisfaction": "Satisfaction au travail", "MaritalStatus": "Situation familiale",
    "MonthlyIncome": "Revenu mensuel", "MonthlyRate": "Taux mensuel",
    "NumCompaniesWorked": "Entreprises précédentes", "OverTime": "Heures supplémentaires",
    "PercentSalaryHike": "Augmentation salariale (%)", "PerformanceRating": "Évaluation de performance",
    "RelationshipSatisfaction": "Satisfaction (relations)", "StockOptionLevel": "Stock-options",
    "TotalWorkingYears": "Années d'expérience", "TrainingTimesLastYear": "Formations l'an dernier",
    "WorkLifeBalance": "Équilibre vie pro/perso", "YearsAtCompany": "Ancienneté",
    "YearsInCurrentRole": "Ancienneté au poste actuel",
    "YearsSinceLastPromotion": "Années depuis la dernière promotion",
    "YearsWithCurrManager": "Années avec le manager actuel"}
VAL_FR = {"Yes": "oui", "No": "non", "Male": "homme", "Female": "femme", "Single": "célibataire",
          "Married": "marié(e)", "Divorced": "divorcé(e)", "Non-Travel": "aucun",
          "Travel_Rarely": "occasionnels", "Travel_Frequently": "fréquents"}
SCALE_1_4 = "Échelle de 1 (faible) à 4 (très élevé)"
HELP = {"EnvironmentSatisfaction": SCALE_1_4, "JobSatisfaction": SCALE_1_4,
        "RelationshipSatisfaction": SCALE_1_4, "JobInvolvement": SCALE_1_4,
        "WorkLifeBalance": "Échelle de 1 (mauvais) à 4 (excellent)",
        "Education": "1 = avant bac … 5 = doctorat", "JobLevel": "1 = débutant … 5 = direction",
        "StockOptionLevel": "0 = aucune … 3 = élevé", "PerformanceRating": "Échelle de 1 à 4"}


def val_fr(v) -> str:
    if isinstance(v, (int, float, np.integer, np.floating)):
        return f"{v:g}"
    return VAL_FR.get(v, str(v))


def split_feat(f: str):
    """'OverTime_Yes' -> ('OverTime', 'Yes') ; 'Age' -> ('Age', None)."""
    if f in META:
        return f, None
    for c in sorted(CAT, key=len, reverse=True):
        if f.startswith(c + "_"):
            return c, f[len(c) + 1:]
    return f, None


VAR_OF = [split_feat(f)[0] for f in FEATS]
GROUP_IDX = {v: [i for i, f in enumerate(VAR_OF) if f == v] for v in RAW_COLS}


def group(M: np.ndarray) -> np.ndarray:
    """Additionne les colonnes encodées d'une même variable d'origine (OverTime_Yes + OverTime_No...)."""
    return np.column_stack([M[:, GROUP_IDX[v]].sum(axis=1) for v in RAW_COLS])


def enc_label(f: str) -> str:
    v, mod = split_feat(f)
    return FR.get(v, v) + (f" = {val_fr(mod)}" if mod is not None else "")


# ----------------------------------------------------------------------- explainers
IS_LINEAR = NAME == "Logistic Regression"
EXPLAINER = shap.LinearExplainer(MODEL, X_TRAIN) if IS_LINEAR else shap.TreeExplainer(MODEL)
BASE_VALUE = float(np.atleast_1d(EXPLAINER.expected_value)[-1])
UNIT = "probabilité" if NAME in ("Random Forest", "Decision Tree") else "log-odds"

CAT_IDX = [i for i, f in enumerate(FEATS) if f not in META]          # colonnes one-hot
LIME = LimeTabularExplainer(X_TRAIN, feature_names=FEATS, class_names=["Reste", "Part"],
                            categorical_features=CAT_IDX, discretize_continuous=True, random_state=42)


def shap_matrix(X_df: pd.DataFrame) -> np.ndarray:
    """Valeurs SHAP de la classe 'départ', quelle que soit la version de shap."""
    sv = EXPLAINER.shap_values(X_df) if IS_LINEAR else EXPLAINER.shap_values(X_df, check_additivity=False)
    if isinstance(sv, list):
        sv = sv[1]
    sv = np.asarray(sv)
    return sv[:, :, 1] if sv.ndim == 3 else sv


@lru_cache(maxsize=1)
def shap_test_grouped() -> np.ndarray:
    return group(shap_matrix(X_TEST_DF))


# ======================================================================= schémas
class EmployeeRequest(BaseModel):
    features: dict[str, Any] = Field(
        ..., description="Caractéristiques de l'employé. Les champs absents prennent la valeur médiane ou la plus fréquente.")
    threshold: float | None = Field(None, ge=0, le=1, description="Seuil de décision (défaut : seuil du modèle).")
    top_k: int = Field(8, ge=1, le=30, description="Nombre de variables retournées dans l'explication.")

    model_config = {"json_schema_extra": {"example": {
        "features": {"Age": 29, "OverTime": "Yes", "MonthlyIncome": 2800, "JobSatisfaction": 1,
                     "BusinessTravel": "Travel_Frequently", "MaritalStatus": "Single"},
        "threshold": 0.5, "top_k": 8}}}


class Contribution(BaseModel):
    feature: str = Field(description="Nom de la variable d'origine")
    label: str = Field(description="Libellé lisible, ex. « Heures supplémentaires = oui »")
    short: str = Field(description="Forme courte pour les phrases de synthèse")
    contribution: float
    direction: str
    sensitive: bool


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
    output_value: float
    other_contribution: float
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
    base_value: float
    output_value: float
    other_contribution: float
    unit: str
    summary: str


class TrainRequest(BaseModel):
    model: Literal["Logistic Regression", "Decision Tree", "KNN", "Random Forest", "XGBoost"]
    params: dict[str, float] = Field(default_factory=dict, description="Hyperparamètres (voir /docs).")

    model_config = {"json_schema_extra": {"example": {"model": "Random Forest",
                                                      "params": {"n_estimators": 200, "max_depth": 8}}}}


# ======================================================================= utilitaires
def to_frame(features: dict) -> tuple[pd.DataFrame, list[str]]:
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


def decision_info(p: float, thr: float) -> dict:
    level = "élevé" if p >= thr else "modéré" if p >= thr / 2 else "faible"
    return dict(probability=round(p, 4), threshold=round(thr, 3), decision=int(p >= thr),
                decision_label="Départ probable" if p >= thr else "Reste probable",
                risk_level=level, model=NAME)


def prepare(req: EmployeeRequest):
    df, ignored = to_frame(req.features)
    x_df = pd.DataFrame(PRE.transform(df), columns=FEATS)
    p = float(MODEL.predict_proba(x_df)[0, 1])
    thr = THRESHOLD if req.threshold is None else req.threshold
    return {**decision_info(p, thr), "ignored_fields": ignored}, x_df, df.iloc[0].to_dict()


def make_contrib(var: str, c: float, raw: dict) -> Contribution:
    name, val = FR.get(var, var), val_fr(raw[var])
    return Contribution(feature=var, label=f"{name} = {val}", short=f"{name.lower()} ({val})",
                        contribution=round(float(c), 4),
                        direction="augmente le risque" if c > 0 else "diminue le risque",
                        sensitive=var in SENSITIVE)


def local_shap(x_df: pd.DataFrame, raw: dict, k: int):
    g = group(shap_matrix(x_df)[0][None, :])[0]
    order = np.argsort(-np.abs(g))[:k]
    items = [make_contrib(RAW_COLS[i], g[i], raw) for i in order]
    other = float(g.sum() - g[order].sum())
    return items, other


def local_lime(x_df: pd.DataFrame, raw: dict, k: int):
    fn = lambda a: MODEL.predict_proba(pd.DataFrame(a, columns=FEATS))
    exp = LIME.explain_instance(x_df.values[0], fn, num_features=min(len(FEATS), 3 * k), num_samples=2000)
    acc: dict[str, float] = {}
    for i, w in exp.local_exp[1]:
        acc[VAR_OF[i]] = acc.get(VAR_OF[i], 0.0) + float(w)
    top = sorted(acc.items(), key=lambda t: -abs(t[1]))[:k]
    return [make_contrib(v, w, raw) for v, w in top]


def sentence(base: dict, items: list[Contribution]) -> str:
    pos = [c.short for c in items if c.contribution > 0][:2]
    neg = [c.short for c in items if c.contribution < 0][:2]
    s = f"Le risque de départ est {base['risk_level']} ({base['probability']:.0%})"
    if base["risk_level"] != "faible":
        s += f", principalement en raison de : {' et '.join(pos)}." if pos else "."
        if neg:
            s += f" Il est atténué par : {neg[0]}."
    else:
        s += f", principalement grâce à : {' et '.join(neg)}." if neg else "."
        if pos:
            s += f" Point de vigilance : {pos[0]}."
    return s


def model_factory(name: str, p: dict):
    i = lambda k, d: int(p.get(k, d))
    if name == "Logistic Regression":
        return LogisticRegression(max_iter=2000, class_weight="balanced", C=float(p.get("C", 1)))
    if name == "Decision Tree":
        return DecisionTreeClassifier(class_weight="balanced", random_state=42,
                                      max_depth=i("max_depth", 5), min_samples_leaf=i("min_samples_leaf", 20))
    if name == "KNN":
        return KNeighborsClassifier(n_neighbors=i("n_neighbors", 11))
    if name == "Random Forest":
        return RandomForestClassifier(class_weight="balanced", random_state=42, n_jobs=1,
                                      n_estimators=min(i("n_estimators", 200), 500), max_depth=i("max_depth", 10))
    return XGBClassifier(scale_pos_weight=SPW, eval_metric="logloss", random_state=42, n_jobs=1,
                         n_estimators=min(i("n_estimators", 200), 500), max_depth=i("max_depth", 3),
                         learning_rate=float(p.get("learning_rate", 0.1)))


# ======================================================================= application
app = FastAPI(
    title="API d'interprétation – Attrition des employés",
    description=("Prédit le risque de départ d'un employé (IBM HR Analytics) et **explique** chaque prédiction "
                 "avec SHAP et LIME. Fournit aussi les explications globales, l'évaluation des modèles et "
                 "un point d'entraînement à la demande. Projet Data Mining – ENSI 2026-2027."),
    version="2.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ----------------------------------------------------------------------- système / modèle
@app.get("/", tags=["Système"], summary="État de l'API")
def health():
    return {"status": "ok", "model": NAME, "docs": "/docs"}


@app.get("/model/info", tags=["Modèle"], summary="Informations sur le modèle déployé")
def model_info():
    return {"model_name": NAME, "best_model": B["best_name"], "threshold": round(THRESHOLD, 3),
            "metrics": B["metrics"], "n_features_raw": len(RAW_COLS), "n_features_encoded": len(FEATS),
            "n_test_samples": len(X_TEST_RAW), "n_flagged_test": int((PROBA_TEST >= THRESHOLD).sum()),
            "shap_unit": UNIT, "sensitive_variables": SENSITIVE}


@app.get("/features", tags=["Modèle"], summary="Variables attendues (libellés, type, bornes, modalités, défaut)")
def features():
    out = []
    for f in B["features_meta"]:
        g = {**f, "label": FR.get(f["name"], f["name"]), "help": HELP.get(f["name"]),
             "sensitive": f["name"] in SENSITIVE}
        if f["type"] == "categorical":
            g["option_labels"] = {o: VAL_FR.get(o, o) for o in f["options"]}
        out.append(g)
    return out


# ----------------------------------------------------------------------- échantillons
@app.get("/samples/ranking", tags=["Échantillons"],
         summary="Employés du jeu de test classés par risque décroissant")
def ranking(limit: int = Query(30, ge=1, le=300)):
    order = np.argsort(-PROBA_TEST)[:limit]
    r = X_TEST_RAW
    return [{"index": int(i), "probability": round(float(PROBA_TEST[i]), 4), "actual": int(Y_TEST[i]),
             "job_role": str(r.loc[i, "JobRole"]), "age": int(r.loc[i, "Age"]),
             "overtime": str(r.loc[i, "OverTime"]), "income": float(r.loc[i, "MonthlyIncome"])}
            for i in order]


@app.get("/samples/{idx}", tags=["Échantillons"], summary="Un employé du jeu de test et ce qui s'est réellement passé")
def sample(idx: int):
    if not 0 <= idx < len(X_TEST_RAW):
        raise HTTPException(404, f"Index hors limites (0 à {len(X_TEST_RAW) - 1}).")
    feats = {k: (v.item() if hasattr(v, "item") else v) for k, v in X_TEST_RAW.iloc[idx].to_dict().items()}
    return {"index": idx, "features": feats, "actual": int(Y_TEST[idx])}


# ----------------------------------------------------------------------- prédiction et explications locales
@app.post("/predict", response_model=PredictResponse, tags=["Prédiction"],
          summary="Probabilité de départ et décision au seuil choisi")
def predict(req: EmployeeRequest):
    base, _, _ = prepare(req)
    return base


@app.post("/explain/shap", response_model=ShapResponse, tags=["Explications locales"],
          summary="Explication SHAP d'un employé (waterfall) + phrase de synthèse")
def explain_shap(req: EmployeeRequest):
    base, x_df, raw = prepare(req)
    items, other = local_shap(x_df, raw, req.top_k)
    return {**base, "base_value": round(BASE_VALUE, 4),
            "output_value": round(BASE_VALUE + sum(c.contribution for c in items) + other, 4),
            "other_contribution": round(other, 4), "unit": UNIT, "contributions": items,
            "summary": sentence(base, items)}


@app.post("/explain/lime", response_model=LimeResponse, tags=["Explications locales"],
          summary="Explication LIME d'un employé")
def explain_lime(req: EmployeeRequest):
    base, x_df, raw = prepare(req)
    return {**base, "contributions": local_lime(x_df, raw, req.top_k)}


@app.post("/explain/compare", response_model=CompareResponse, tags=["Explications locales"],
          summary="SHAP et LIME côte à côte, avec mesure d'accord")
def explain_compare(req: EmployeeRequest):
    base, x_df, raw = prepare(req)
    s_items, other = local_shap(x_df, raw, req.top_k)
    l_items = local_lime(x_df, raw, req.top_k)
    s_map = {c.feature: c.contribution for c in s_items}
    l_map = {c.feature: c.contribution for c in l_items}
    common = sorted(set(s_map) & set(l_map))
    same = [f for f in common if np.sign(s_map[f]) == np.sign(l_map[f])]
    return {**base, "shap": s_items, "lime": l_items, "common_features": common,
            "overlap_ratio": round(len(common) / req.top_k, 2),
            "same_sign_ratio": round(len(same) / len(common), 2) if common else 0.0,
            "base_value": round(BASE_VALUE, 4),
            "output_value": round(BASE_VALUE + sum(c.contribution for c in s_items) + other, 4),
            "other_contribution": round(other, 4), "unit": UNIT, "summary": sentence(base, s_items)}


# ----------------------------------------------------------------------- explications globales
@app.get("/explain/global", tags=["Explications globales"],
         summary="Importance SHAP, importance native de chaque modèle et importance par permutation (tous modèles)")
def explain_global(top_k: int = Query(15, ge=1, le=60)):
    mean_abs = np.abs(shap_test_grouped()).mean(axis=0)
    shap_rows = sorted(({"feature": v, "label": FR.get(v, v), "shap_mean_abs": round(float(s), 4),
                         "sensitive": v in SENSITIVE} for v, s in zip(RAW_COLS, mean_abs)),
                       key=lambda r: -r["shap_mean_abs"])
    native = {m: sorted(({"label": enc_label(f), "value": x} for f, x in d.items()),
                        key=lambda r: -abs(r["value"]))[:top_k]
              for m, d in B["native_importance"].items()}
    perm = {m: [{"feature": v, "label": FR.get(v, v), "importance": x, "sensitive": v in SENSITIVE}
                for v, x in d.items()] for m, d in B["perm_importance"].items()}
    return {"model": NAME, "shap": shap_rows[:top_k], "native": native, "native_kind": B["native_kind"],
            "permutation": perm}


@app.get("/explain/shap-values", tags=["Explications globales"],
         summary="Valeurs SHAP du jeu de test par variable (beeswarm et graphiques de dépendance)")
def shap_values_global(top_k: int = Query(12, ge=1, le=60)):
    G = shap_test_grouped()
    order = np.argsort(-np.abs(G).mean(axis=0))[:top_k]
    out = []
    for i in order:
        v = RAW_COLS[i]
        col = X_TEST_RAW[v]
        vals = col.astype(float).tolist() if META[v]["type"] == "numeric" else [VAL_FR.get(x, x) for x in col]
        out.append({"feature": v, "label": FR.get(v, v), "type": META[v]["type"], "sensitive": v in SENSITIVE,
                    "shap": np.round(G[:, i], 4).tolist(), "values": vals})
    return {"model": NAME, "unit": UNIT, "n": int(len(G)), "variables": out}


@lru_cache(maxsize=4)
def _agreement(n: int) -> dict:
    rows = []
    for i in np.argsort(-PROBA_TEST)[:n]:
        i = int(i)
        raw, x_df = X_TEST_RAW.iloc[i].to_dict(), X_TEST_DF.iloc[[i]]
        s = {c.feature: c.contribution for c in local_shap(x_df, raw, 5)[0]}
        l = {c.feature: c.contribution for c in local_lime(x_df, raw, 5)}
        common = set(s) & set(l)
        same = [f for f in common if np.sign(s[f]) == np.sign(l[f])]
        rows.append({"index": i, "overlap": len(common) / 5,
                     "same_sign": len(same) / len(common) if common else 0.0})
    return {"n": n, "mean_overlap": round(float(np.mean([r["overlap"] for r in rows])), 3),
            "mean_same_sign": round(float(np.mean([r["same_sign"] for r in rows])), 3), "details": rows}


@app.get("/explain/agreement", tags=["Explications globales"],
         summary="Accord SHAP / LIME mesuré sur les n employés les plus à risque (top 5 variables)")
def agreement(n: int = Query(15, ge=3, le=40)):
    return _agreement(n)


# ----------------------------------------------------------------------- modèles
@app.get("/models/evaluation", tags=["Modèles"],
         summary="Probabilités test de tous les modèles, tableau comparatif et hyperparamètres retenus")
def evaluation():
    return {"y_test": Y_TEST.tolist(), "proba": B["eval_proba"], "results": B["results_records"],
            "tuning": B["tuning_records"], "best_model": B["best_name"], "default_threshold": round(THRESHOLD, 3)}


@app.post("/models/train", tags=["Modèles"],
          summary="Entraîne un modèle avec des hyperparamètres choisis et renvoie ses probabilités sur le jeu de test")
def train(req: TrainRequest):
    t0 = time.time()
    m = model_factory(req.model, req.params).fit(pd.DataFrame(X_TRAIN, columns=FEATS), Y_TRAIN)
    p = m.predict_proba(X_TEST_DF)[:, 1]
    return {"model": req.model, "params": req.params, "proba": np.round(p, 5).tolist(),
            "seconds": round(time.time() - t0, 1)}