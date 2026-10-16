"""HR Analytics – Prévention des départs d'employés (application Streamlit)."""
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import seaborn as sns
import streamlit as st
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score, roc_curve)

st.set_page_config(page_title="HR Analytics – Départs", page_icon="👥", layout="wide")
sns.set_theme(style="whitegrid")
RED, GREEN, BLUE, GREY = "#D1495B", "#4C9F70", "#4C78A8", "#9aa0a6"
CSV_NAME = "WA_Fn-UseC_-HR-Employee-Attrition.csv"

KEY_FEATURES = ["OverTime", "JobSatisfaction", "MonthlyIncome", "Age", "BusinessTravel", "WorkLifeBalance",
                "MaritalStatus", "YearsSinceLastPromotion", "JobRole", "YearsAtCompany"]

ACTIONS = {
    "OverTime": "Revoir la charge de travail : limiter ou compenser les heures supplémentaires.",
    "MonthlyIncome": "Comparer la rémunération au marché et aux collègues de même niveau.",
    "JobSatisfaction": "Organiser un entretien individuel pour identifier les sources d'insatisfaction.",
    "EnvironmentSatisfaction": "Échanger sur les conditions de travail (équipe, outils, environnement).",
    "WorkLifeBalance": "Proposer plus de flexibilité (télétravail, horaires aménagés).",
    "BusinessTravel": "Réduire ou mieux répartir les déplacements professionnels.",
    "YearsSinceLastPromotion": "Discuter des perspectives d'évolution et d'un calendrier de promotion.",
    "StockOptionLevel": "Étudier un dispositif d'intéressement ou de fidélisation (primes, stock-options).",
    "DistanceFromHome": "Proposer du télétravail ou une aide au transport.",
    "JobInvolvement": "Confier des missions plus engageantes ou davantage de responsabilités.",
    "TrainingTimesLastYear": "Proposer un plan de formation.",
    "YearsWithCurrManager": "Prévoir un point avec le manager sur la relation de travail.",
    "RelationshipSatisfaction": "Favoriser la cohésion d'équipe (mentorat, temps collectifs).",
    "YearsInCurrentRole": "Étudier une mobilité interne ou un changement de mission.",
    "JobLevel": "Clarifier le parcours de carrière et les critères de progression.",
    "PercentSalaryHike": "Réexaminer la politique d'augmentation salariale.",
}


# ======================================================================= accès à l'API
def get_api_url() -> str:
    try:
        return st.secrets["API_URL"]
    except Exception:
        return os.getenv("API_URL", "http://127.0.0.1:8000")


API_URL = get_api_url().rstrip("/")


@st.cache_data(ttl=3600, show_spinner="Analyse en cours…")
def api_get(path: str):
    r = requests.get(f"{API_URL}{path}", timeout=120)
    r.raise_for_status()
    return r.json()


def api_post(path: str, payload: dict):
    r = requests.post(f"{API_URL}{path}", json=payload, timeout=120)
    if r.status_code != 200:
        raise RuntimeError(r.json().get("detail", r.text))
    return r.json()


@st.cache_data(show_spinner="Chargement des données…")
def load_data() -> pd.DataFrame:
    try:
        import kagglehub
        path = kagglehub.dataset_download("pavansubhasht/ibm-hr-analytics-attrition-dataset")
        return pd.read_csv(os.path.join(path, CSV_NAME))
    except Exception:
        return pd.read_csv(Path(__file__).resolve().parent.parent / "data" / CSV_NAME)


try:
    INFO = api_get("/model/info")
    META = api_get("/features")
except Exception:
    st.error("Le service d'analyse est momentanément indisponible. S'il vient d'être réveillé, "
             "patientez environ une minute puis rechargez la page.")
    st.stop()

LABEL = {f["name"]: f["label"] for f in META}
SENSITIVE = set(INFO["sensitive_variables"])


# ======================================================================= graphiques
def show(fig):
    st.pyplot(fig)
    plt.close(fig)


def bar_contribs(items, title):
    d = pd.DataFrame(items).sort_values("contribution")
    fig, ax = plt.subplots(figsize=(6.5, 0.45 * len(d) + 1.4))
    ax.barh(d["label"], d["contribution"], color=[RED if v > 0 else GREEN for v in d["contribution"]])
    ax.axvline(0, color="black", lw=0.8)
    ax.set_title(title)
    ax.set_xlabel("Contribution (rouge : augmente le risque, vert : le diminue)")
    plt.tight_layout()
    return fig


def waterfall(res):
    steps = [(c["label"], c["contribution"]) for c in res["shap"]]
    if abs(res["other_contribution"]) > 1e-6:
        steps.append(("Autres facteurs", res["other_contribution"]))
    fig, ax = plt.subplots(figsize=(8, 0.5 * len(steps) + 1.8))
    cum = res["base_value"]
    for i, (_, v) in enumerate(steps):
        ax.barh(i, v, left=cum, color=RED if v > 0 else GREEN)
        ax.annotate(f"{v:+.2f}", (cum + v, i), xytext=(4 if v >= 0 else -4, 0), textcoords="offset points",
                    ha="left" if v >= 0 else "right", va="center", fontsize=8)
        cum += v
    ax.set_yticks(range(len(steps)))
    ax.set_yticklabels([s[0] for s in steps])
    ax.invert_yaxis()
    ax.axvline(res["base_value"], color=GREY, ls=":", label="Employé « moyen »")
    ax.axvline(cum, color="black", ls="--", label="Cet employé")
    ax.margins(x=0.15)
    ax.set_xlabel(f"Score du modèle ({res['unit']})")
    ax.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    return fig


def gauge(p, thr):
    fig, ax = plt.subplots(figsize=(8, 1.2))
    ax.barh([0], [1], color="#e6e6e6")
    ax.barh([0], [p], color=RED if p >= thr else GREEN)
    ax.axvline(thr, color="black", lw=2)
    ax.text(thr, 0.62, f"seuil {thr:.0%}", ha="center", fontsize=8)
    ax.set(xlim=(0, 1), ylim=(-0.5, 0.95), yticks=[])
    ax.set_xticks([0, .25, .5, .75, 1])
    ax.set_xticklabels(["0 %", "25 %", "50 %", "75 %", "100 %"])
    ax.grid(False)
    sns.despine(left=True)
    plt.tight_layout()
    return fig


def beeswarm(sv):
    vars_ = sv["variables"][::-1]
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(9, 0.5 * len(vars_) + 1.8))
    sc = None
    for i, v in enumerate(vars_):
        s = np.array(v["shap"])
        hist, edges = np.histogram(s, bins=30)
        dens = hist[np.clip(np.digitize(s, edges) - 1, 0, 29)] / hist.max()
        y = i + rng.uniform(-1, 1, len(s)) * 0.38 * dens
        if v["type"] == "numeric":
            x = np.array(v["values"], float)
            rank = (x - x.min()) / (x.max() - x.min() + 1e-9)
            sc = ax.scatter(s, y, c=rank, cmap="coolwarm", vmin=0, vmax=1, s=12, alpha=0.85)
        else:
            ax.scatter(s, y, color=GREY, s=12, alpha=0.7)
    ax.set_yticks(range(len(vars_)))
    ax.set_yticklabels([v["label"] for v in vars_])
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel(f"Impact sur le risque de départ ({sv['unit']}) – vers la droite : augmente le risque")
    if sc is not None:
        cb = fig.colorbar(sc, ax=ax, pad=0.01)
        cb.set_ticks([0, 1])
        cb.set_ticklabels(["faible", "élevée"])
        cb.set_label("Valeur de la variable (gris : variable catégorielle)")
    plt.tight_layout()
    return fig


def dependence(v):
    s = np.array(v["shap"])
    fig, ax = plt.subplots(figsize=(7, 4))
    if v["type"] == "numeric":
        x = np.array(v["values"], float)
        ax.scatter(x, s, s=14, alpha=0.55, color=BLUE)
        if len(np.unique(x)) > 2:
            b = pd.qcut(x, 8, duplicates="drop")
            m = pd.DataFrame({"x": x, "s": s, "b": b}).groupby("b", observed=True).mean()
            ax.plot(m["x"], m["s"], color=RED, lw=2, label="Tendance moyenne")
            ax.legend()
        ax.axhline(0, color="black", lw=0.8)
        ax.set(xlabel=v["label"], ylabel="Impact sur le risque (SHAP)")
    else:
        d = pd.DataFrame({"cat": v["values"], "shap": s})
        order = d.groupby("cat")["shap"].mean().sort_values().index
        sns.boxplot(data=d, y="cat", x="shap", order=order, color="#c7d7ea", fliersize=0, ax=ax)
        sns.stripplot(data=d, y="cat", x="shap", order=order, size=3, color=BLUE, alpha=0.5, ax=ax)
        ax.axvline(0, color="black", lw=0.8)
        ax.set(xlabel="Impact sur le risque (SHAP)", ylabel="")
    ax.set_title(f"Dépendance : {v['label']}")
    plt.tight_layout()
    return fig


def rate_table(d, col, ybin):
    s = d[col]
    binned = pd.api.types.is_numeric_dtype(s) and s.nunique() > 8
    if binned:
        s = pd.qcut(s, 5, duplicates="drop")
    t = ybin.groupby(s, observed=True).agg(taux="mean", n="size")
    t.index = t.index.astype(str)
    t = t.reset_index(names="groupe")
    return t if binned else t.sort_values("taux", ascending=False)


def rate_chart(t, title, overall):
    fig, ax = plt.subplots(figsize=(8, 0.5 * len(t) + 1.6))
    sns.barplot(data=t, x=t["taux"] * 100, y="groupe", color=RED, ax=ax)
    ax.bar_label(ax.containers[0], fmt="%.1f%%", padding=3)
    ax.axvline(overall * 100, color="black", ls="--", lw=1, label=f"Moyenne ({overall:.1%})")
    ax.set(title=title, xlabel="Taux de départ (%)", ylabel="")
    ax.legend(loc="lower right")
    plt.tight_layout()
    return fig


# ======================================================================= pages
def page_dashboard():
    st.title("Prévention des départs d'employés")
    st.caption("Repérer les employés à risque, comprendre pourquoi, et agir à temps.")
    df = load_data()
    c = st.columns(4)
    c[0].metric("Effectif analysé", f"{len(df):,}".replace(",", " "))
    c[1].metric("Taux de départ observé", f"{(df['Attrition'] == 'Yes').mean():.1%}")
    c[2].metric("Départs détectés (rappel)", f"{INFO['metrics']['Rappel']:.0%}")
    c[3].metric("Fiabilité du modèle (AUC)", f"{INFO['metrics']['AUC-ROC']:.2f}")

    left, right = st.columns(2)
    with left:
        st.subheader("Ce qui pèse le plus sur les départs")
        g = api_get("/explain/global?top_k=8")
        d = pd.DataFrame(g["shap"]).sort_values("shap_mean_abs")
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.barh(d["label"], d["shap_mean_abs"], color=[GREY if s else BLUE for s in d["sensitive"]])
        ax.set_xlabel("Poids moyen dans les prédictions (SHAP)")
        plt.tight_layout()
        show(fig)
        st.caption("Gris : variable sensible (âge, genre, situation familiale) – à interpréter avec prudence.")
    with right:
        st.subheader("Employés à surveiller en priorité")
        r = pd.DataFrame(api_get("/samples/ranking?limit=10"))
        r["Risque (%)"] = (r["probability"] * 100).round(0)
        r["Heures sup."] = r["overtime"].map({"Yes": "oui", "No": "non"})
        st.dataframe(
            r.rename(columns={"index": "N°", "job_role": "Poste", "age": "Âge", "income": "Revenu"})
             [["N°", "Poste", "Âge", "Heures sup.", "Revenu", "Risque (%)"]],
            hide_index=True,
            column_config={"Risque (%)": st.column_config.ProgressColumn("Risque de départ", min_value=0,
                                                                         max_value=100, format="%.0f%%")})
        st.caption("Pour comprendre le risque d'un employé, ouvrez « Prédire un employé ».")
    st.info("Le score est une **aide à la décision** : il sert à ouvrir une discussion avec l'employé, "
            "jamais à le sanctionner ou à restreindre ses opportunités.")


def page_analysis():
    st.title("Analyse des départs")
    df = load_data()
    y = (df["Attrition"] == "Yes").astype(int)
    overall = y.mean()
    skip = {"Attrition", "EmployeeCount", "StandardHours", "Over18", "EmployeeNumber"}
    cols = [c for c in df.columns if c not in skip]
    nums = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    t1, t2, t3, t4, t5 = st.tabs(["Vue d'ensemble", "Où se concentrent les départs", "Profils",
                                  "Corrélations", "Importer mon fichier"])
    with t1:
        c = st.columns(4)
        c[0].metric("Employés", len(df))
        c[1].metric("Variables", df.shape[1])
        c[2].metric("Valeurs manquantes", int(df.isna().sum().sum()))
        c[3].metric("Ont quitté l'entreprise", f"{overall:.1%}")
        a, b = st.columns([1, 2])
        with a:
            fig, ax = plt.subplots(figsize=(4, 3.5))
            cnt = df["Attrition"].value_counts().reindex(["No", "Yes"])
            ax.bar(["Restés", "Partis"], cnt.values, color=[GREEN, RED])
            for i, v in enumerate(cnt.values):
                ax.text(i, v + 10, f"{v} ({v / len(df):.1%})", ha="center")
            ax.set_title("Répartition des départs")
            show(fig)
        with b:
            st.dataframe(df.head(8))
        with st.expander("Statistiques descriptives"):
            st.dataframe(df[nums].describe().T.round(2))
        miss = df.isna().sum()
        with st.expander("Valeurs manquantes"):
            st.write("Aucune valeur manquante." if miss.sum() == 0 else miss[miss > 0].to_frame("manquantes"))
    with t2:
        col = st.selectbox("Variable", cols, format_func=lambda c: LABEL.get(c, c), key="rate_var")
        show(rate_chart(rate_table(df, col, y), f"Taux de départ selon : {LABEL.get(col, col)}", overall))
    with t3:
        col = st.selectbox("Variable numérique", nums, format_func=lambda c: LABEL.get(c, c), key="dist_var")
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        sns.histplot(data=df, x=col, hue="Attrition", palette={"No": GREEN, "Yes": RED}, kde=True, stat="density",
                     common_norm=False, element="step", ax=axes[0])
        sns.boxplot(data=df, x="Attrition", y=col, hue="Attrition", palette={"No": GREEN, "Yes": RED},
                    order=["No", "Yes"], legend=False, ax=axes[1])
        axes[0].set_xlabel(LABEL.get(col, col))
        axes[1].set_ylabel(LABEL.get(col, col))
        plt.tight_layout()
        show(fig)
    with t4:
        corr = df[nums].corr()
        a, b = st.columns(2)
        with a:
            fig, ax = plt.subplots(figsize=(7, 6))
            sns.heatmap(corr, mask=np.triu(np.ones_like(corr, dtype=bool)), cmap="coolwarm", center=0,
                        vmin=-1, vmax=1, square=True, cbar_kws={"shrink": 0.7}, ax=ax)
            ax.set_title("Corrélations entre variables numériques")
            show(fig)
        with b:
            ct = df[nums].assign(Attrition=y).corr()["Attrition"].drop("Attrition").sort_values()
            fig, ax = plt.subplots(figsize=(7, 6))
            ax.barh([LABEL.get(c, c) for c in ct.index], ct.values, color=[RED if v > 0 else GREEN for v in ct])
            ax.set_title("Corrélation avec le départ")
            plt.tight_layout()
            show(fig)
    with t5:
        up = st.file_uploader("Fichier CSV", type="csv")
        if up is not None:
            d = pd.read_csv(up)
            target = st.selectbox("Variable cible", d.columns)
            pos = st.selectbox("Valeur de la classe d'intérêt", sorted(d[target].dropna().unique().tolist(), key=str))
            yb = (d[target] == pos).astype(int)
            st.write(f"{len(d)} lignes · {d.shape[1]} colonnes · {int(d.isna().sum().sum())} valeurs manquantes · "
                     f"classe d'intérêt : **{yb.mean():.1%}**")
            col = st.selectbox("Taux de la classe d'intérêt par variable", [c for c in d.columns if c != target])
            show(rate_chart(rate_table(d, col, yb), f"Taux par {col}", yb.mean()))


def field(f, default, key):
    if f["type"] == "categorical":
        opts, lab = f["options"], f["option_labels"]
        return st.selectbox(f["label"], opts, index=opts.index(default) if default in opts else 0,
                            format_func=lambda o: lab.get(o, o), key=key, help=f.get("help"))
    if f.get("integer"):
        return st.number_input(f["label"], int(f["min"]), int(f["max"]), int(default), step=1, key=key,
                               help=f.get("help"))
    return st.number_input(f["label"], float(f["min"]), float(f["max"]), float(default), key=key, help=f.get("help"))


def page_predict():
    st.title("Prédire et expliquer le risque d'un employé")
    src = st.radio("Quel employé analyser ?", ["Saisir un employé", "Choisir un employé existant"], horizontal=True)
    defaults = {f["name"]: f["default"] for f in META}
    tag = "new"
    if src == "Choisir un employé existant":
        rank = {r["index"]: r for r in api_get("/samples/ranking?limit=100")}
        idx = st.selectbox("Employé (classés du plus au moins à risque)", list(rank),
                           format_func=lambda i: f"N°{i} · {rank[i]['job_role']}, {rank[i]['age']} ans · "
                                                 f"risque estimé {rank[i]['probability']:.0%}")
        s = api_get(f"/samples/{idx}")
        defaults.update(s["features"])
        tag = f"emp{idx}"
        st.caption("Dans la réalité, cet employé " + ("**a quitté** l'entreprise." if s["actual"] else "**est resté**."))

    by_name = {f["name"]: f for f in META}
    values = {}
    st.markdown("**Caractéristiques principales**")
    cols = st.columns(3)
    for i, n in enumerate([k for k in KEY_FEATURES if k in by_name]):
        with cols[i % 3]:
            values[n] = field(by_name[n], defaults[n], f"{n}_{tag}")
    with st.expander("Autres caractéristiques"):
        cols = st.columns(3)
        for i, n in enumerate([f["name"] for f in META if f["name"] not in values]):
            with cols[i % 3]:
                values[n] = field(by_name[n], defaults[n], f"{n}_{tag}")
    with st.expander("Réglages de l'analyse"):
        thr = st.slider("Seuil de décision : au-delà, l'employé est signalé « à risque »", 0.05, 0.95,
                        float(INFO["threshold"]), 0.01)
        top_k = st.slider("Nombre de facteurs affichés", 4, 12, 8)

    if st.button("Analyser cet employé", type="primary"):
        try:
            with st.spinner("Calcul des explications…"):
                st.session_state["res"] = api_post("/explain/compare",
                                                   {"features": values, "threshold": thr, "top_k": top_k})
        except Exception as e:
            st.error(f"Analyse impossible : {e}")
    res = st.session_state.get("res")
    if not res:
        return

    st.divider()
    m = st.columns(3)
    m[0].metric("Probabilité de départ", f"{res['probability']:.0%}")
    m[1].metric("Décision", res["decision_label"])
    m[2].metric("Niveau de risque", res["risk_level"].capitalize())
    show(gauge(res["probability"], res["threshold"]))
    (st.warning if res["decision"] else st.success)(res["summary"])

    t1, t2, t3 = st.tabs(["Pourquoi cette prédiction ?", "Seconde opinion (LIME)", "Que faire ?"])
    with t1:
        st.caption("Chaque barre montre de combien une caractéristique pousse le risque vers le haut (rouge) "
                   "ou vers le bas (vert) par rapport à un employé moyen.")
        show(waterfall(res))
    with t2:
        a, b = st.columns(2)
        with a:
            show(bar_contribs(res["shap"], "Méthode 1 : SHAP"))
        with b:
            show(bar_contribs(res["lime"], "Méthode 2 : LIME"))
        lab = {c["feature"]: c["label"] for c in res["shap"] + res["lime"]}
        s = pd.DataFrame(res["shap"])[["feature", "contribution"]].rename(columns={"contribution": "SHAP"})
        l = pd.DataFrame(res["lime"])[["feature", "contribution"]].rename(columns={"contribution": "LIME"})
        cmp_ = s.merge(l, on="feature", how="outer")
        cmp_.insert(0, "Variable", cmp_["feature"].map(lab))
        cmp_["Même sens ?"] = np.where(cmp_[["SHAP", "LIME"]].notna().all(axis=1),
                                       np.where(np.sign(cmp_["SHAP"]) == np.sign(cmp_["LIME"]), "oui", "non"), "—")
        st.dataframe(cmp_.drop(columns="feature").round(3), hide_index=True)
        st.write(f"**Accord entre les deux méthodes** : {res['overlap_ratio']:.0%} de facteurs en commun, "
                 f"{res['same_sign_ratio']:.0%} d'entre eux dans le même sens.")
    with t3:
        pos = [c for c in res["shap"] if c["contribution"] > 0]
        todo = [c for c in pos if not c["sensitive"] and c["feature"] in ACTIONS]
        if todo:
            for c in todo:
                st.markdown(f"- **{c['label']}** → {ACTIONS[c['feature']]}")
        else:
            st.write("Aucun facteur actionnable ne pèse nettement sur ce risque.")
        sens = [c["label"] for c in pos if c["sensitive"]]
        if sens:
            st.warning("Facteurs sensibles qui jouent dans le score : " + ", ".join(sens) + ". "
                       "Ils ne sont pas actionnables et ne doivent jamais fonder une décision RH.")


def page_models():
    st.title("Performance des modèles")
    ev = api_get("/models/evaluation")
    y = np.array(ev["y_test"])
    proba = {k: np.array(v) for k, v in ev["proba"].items()}
    thr = st.slider("Seuil de décision (effet en direct sur la précision et le rappel)", 0.05, 0.95,
                    float(ev["default_threshold"]), 0.01,
                    help="Seuil bas : on détecte plus de départs mais on se trompe plus souvent. "
                         "Seuil haut : on se trompe moins mais on en rate davantage.")
    rows = []
    for n, p in proba.items():
        pred = (p >= thr).astype(int)
        rows.append({"Modèle": n, "Précision": precision_score(y, pred, zero_division=0),
                     "Rappel": recall_score(y, pred), "F1": f1_score(y, pred),
                     "AUC-ROC": roc_auc_score(y, p), "PR-AUC": average_precision_score(y, p),
                     "Employés signalés": int(pred.sum())})
    tab = pd.DataFrame(rows).set_index("Modèle").sort_values("AUC-ROC", ascending=False)
    st.dataframe(tab.style.background_gradient(cmap="YlGnBu", subset=["Précision", "Rappel", "F1", "AUC-ROC", "PR-AUC"])
                 .format({c: "{:.3f}" for c in ["Précision", "Rappel", "F1", "AUC-ROC", "PR-AUC"]}))
    st.caption(f"Meilleur modèle (AUC-ROC) : **{tab.index[0]}** · Modèle utilisé pour les explications : "
               f"**{INFO['model_name']}** · Taux de départ du jeu de test : {y.mean():.1%}")

    t1, t2, t3, t4 = st.tabs(["Courbes ROC et précision-rappel", "Matrice de confusion",
                              "Réglages retenus", "Tester un autre réglage"])
    with t1:
        colors = dict(zip(proba, sns.color_palette("tab10", len(proba))))
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        for n, p in proba.items():
            fpr, tpr, _ = roc_curve(y, p)
            pr, rc, _ = precision_recall_curve(y, p)
            axes[0].plot(fpr, tpr, color=colors[n], lw=2, label=f"{n} ({roc_auc_score(y, p):.3f})")
            axes[1].plot(rc, pr, color=colors[n], lw=2, label=f"{n} ({average_precision_score(y, p):.3f})")
            pred = p >= thr
            axes[0].scatter(pred[y == 0].mean(), pred[y == 1].mean(), color=colors[n], s=60, edgecolor="black", zorder=5)
            axes[1].scatter(recall_score(y, pred), precision_score(y, pred, zero_division=0), color=colors[n], s=60,
                            edgecolor="black", zorder=5)
        axes[0].plot([0, 1], [0, 1], "k--", lw=1)
        axes[1].axhline(y.mean(), color="k", ls="--", lw=1)
        axes[0].set(title="Courbes ROC", xlabel="Taux de fausses alertes", ylabel="Taux de départs détectés")
        axes[1].set(title="Courbes précision-rappel", xlabel="Rappel", ylabel="Précision")
        axes[0].legend(loc="lower right")
        axes[1].legend(loc="upper right")
        plt.tight_layout()
        show(fig)
        st.caption("Les points entourés indiquent la position de chaque modèle au seuil choisi.")
    with t2:
        n = st.selectbox("Modèle", list(proba), index=list(proba).index(tab.index[0]))
        pred = (proba[n] >= thr).astype(int)
        a, b = st.columns([1, 1])
        with a:
            fig, ax = plt.subplots(figsize=(4.5, 3.8))
            sns.heatmap(confusion_matrix(y, pred), annot=True, fmt="d", cmap="Blues", cbar=False,
                        xticklabels=["Reste", "Part"], yticklabels=["Reste", "Part"], ax=ax)
            ax.set(xlabel="Prédit", ylabel="Réel", title=n)
            show(fig)
        with b:
            tp, fn = int(((pred == 1) & (y == 1)).sum()), int(((pred == 0) & (y == 1)).sum())
            fp = int(((pred == 1) & (y == 0)).sum())
            st.write(f"Sur **{int(y.sum())}** départs réels, **{tp}** sont détectés et **{fn}** manqués.")
            st.write(f"**{fp}** employés signalés à tort (fausses alertes).")
    with t3:
        st.dataframe(pd.DataFrame(ev["tuning"]).set_index("Modèle"))
    with t4:
        st.caption("Entraînez un modèle avec vos propres hyperparamètres et comparez-le à la version retenue.")
        mdl = st.selectbox("Modèle à entraîner", list(proba), key="train_model")
        p = {}
        if mdl == "Logistic Regression":
            p["C"] = st.select_slider("C (régularisation)", [0.01, 0.1, 1.0, 10.0, 100.0], 1.0)
        elif mdl == "Decision Tree":
            p["max_depth"] = st.slider("Profondeur maximale", 2, 20, 5)
            p["min_samples_leaf"] = st.slider("Taille minimale d'une feuille", 1, 50, 20)
        elif mdl == "KNN":
            p["n_neighbors"] = st.slider("Nombre de voisins", 3, 51, 11, step=2)
        elif mdl == "Random Forest":
            p["n_estimators"] = st.slider("Nombre d'arbres", 50, 500, 200, step=50)
            p["max_depth"] = st.slider("Profondeur maximale", 3, 25, 10)
        else:
            p["n_estimators"] = st.slider("Nombre d'arbres", 50, 500, 200, step=50)
            p["max_depth"] = st.slider("Profondeur maximale", 2, 10, 3)
            p["learning_rate"] = st.select_slider("Taux d'apprentissage", [0.01, 0.05, 0.1, 0.3], 0.1)
        if st.button("Entraîner"):
            with st.spinner("Entraînement en cours…"):
                try:
                    out = api_post("/models/train", {"model": mdl, "params": p})
                except Exception as e:
                    st.error(f"Entraînement impossible : {e}")
                    st.stop()
            q = np.array(out["proba"])
            pr_ = (q >= thr).astype(int)
            new = {"Version": "Votre réglage", "Précision": precision_score(y, pr_, zero_division=0),
                   "Rappel": recall_score(y, pr_), "F1": f1_score(y, pr_),
                   "AUC-ROC": roc_auc_score(y, q), "PR-AUC": average_precision_score(y, q)}
            old = {"Version": "Réglage retenu", **tab.loc[mdl, ["Précision", "Rappel", "F1", "AUC-ROC", "PR-AUC"]].to_dict()}
            st.dataframe(pd.DataFrame([old, new]).set_index("Version").round(3))
            st.caption(f"Entraînement terminé en {out['seconds']} s.")


def page_factors():
    st.title("Facteurs de risque")
    g = api_get("/explain/global?top_k=15")
    sv = api_get("/explain/shap-values?top_k=12")
    t1, t2, t3, t4 = st.tabs(["Poids propres au modèle", "Comparaison entre modèles",
                              "Impact et sens (SHAP)", "Détail par variable"])
    with t1:
        mdl = st.selectbox("Modèle", list(g["native"]))
        d = pd.DataFrame(g["native"][mdl]).sort_values("value")
        fig, ax = plt.subplots(figsize=(8, 0.4 * len(d) + 1.5))
        signed = mdl == "Logistic Regression"
        ax.barh(d["label"], d["value"], color=[(RED if v > 0 else GREEN) if signed else BLUE for v in d["value"]])
        ax.axvline(0, color="black", lw=0.8)
        ax.set_title(g["native_kind"][mdl])
        plt.tight_layout()
        show(fig)
        st.caption("Rouge : augmente le risque de départ · Vert : le diminue." if signed else
                   "Plus la barre est longue, plus la variable est utilisée par le modèle (sans indiquer le sens).")
        st.caption("Le modèle des k plus proches voisins n'a pas d'importance propre : voir l'onglet suivant.")
    with t2:
        rows = [{"Variable": r["label"], "Modèle": m, "Importance": r["importance"]}
                for m, lst in g["permutation"].items() for r in lst]
        pv = pd.DataFrame(rows).pivot(index="Variable", columns="Modèle", values="Importance")
        pv = pv.loc[pv.mean(axis=1).sort_values(ascending=False).index[:12]]
        fig, ax = plt.subplots(figsize=(9, 5.5))
        sns.heatmap(pv, annot=True, fmt=".3f", cmap="YlOrRd", ax=ax, cbar_kws={"label": "Baisse d'AUC"})
        ax.set_title("Importance par permutation (baisse d'AUC quand la variable est mélangée)")
        ax.set_ylabel("")
        plt.tight_layout()
        show(fig)
        st.caption("Cette mesure est comparable d'un modèle à l'autre : une variable importante pour tous les "
                   "modèles est un facteur de départ robuste.")
    with t3:
        show(beeswarm(sv))
        st.caption("Chaque point est un employé. À droite de la ligne centrale, la variable augmente son risque ; "
                   "à gauche, elle le diminue. Pour les variables numériques, la couleur indique si la valeur est "
                   "faible (bleu) ou élevée (rouge).")
    with t4:
        v = st.selectbox("Variable", sv["variables"], format_func=lambda v: v["label"])
        show(dependence(v))


def page_fairness():
    st.title("Équité et fiabilité du modèle")
    g = api_get("/explain/global?top_k=60")
    sv = api_get("/explain/shap-values?top_k=60")
    ev = api_get("/models/evaluation")

    st.subheader("1. Le modèle s'appuie-t-il sur des variables sensibles ?")
    d = pd.DataFrame(g["shap"])
    d["rang"] = np.arange(1, len(d) + 1)
    d["part"] = d["shap_mean_abs"] / d["shap_mean_abs"].sum()
    sens = d[d["sensitive"]]
    st.metric("Part de l'influence portée par âge, genre et situation familiale", f"{sens['part'].sum():.0%}")
    st.dataframe(sens.assign(part=(sens["part"] * 100).round(1))
                 .rename(columns={"label": "Variable sensible", "rang": "Rang", "part": "Part de l'influence (%)"})
                 [["Variable sensible", "Rang", "Part de l'influence (%)"]], hide_index=True)
    v = st.selectbox("Effet de la variable sensible", [x for x in sv["variables"] if x["sensitive"]],
                     format_func=lambda x: x["label"])
    show(dependence(v))
    st.caption("Un effet marqué signifie que le score diffère selon le groupe : risque de discrimination indirecte. "
               "Ces variables ne doivent jamais servir à décider d'une promotion, d'une augmentation ou d'un licenciement.")

    st.subheader("2. Les explications SHAP et LIME sont-elles cohérentes ?")
    n = st.slider("Nombre d'employés à risque testés", 5, 30, 15)
    ag = api_get(f"/explain/agreement?n={n}")
    a, b = st.columns(2)
    a.metric("Variables en commun (top 5)", f"{ag['mean_overlap']:.0%}")
    b.metric("Même sens d'effet", f"{ag['mean_same_sign']:.0%}")
    st.caption("Plus ces taux sont élevés, plus les deux méthodes indépendantes racontent la même histoire.")

    st.subheader("3. Performance ou lisibilité ?")
    res = pd.DataFrame(ev["results"]).set_index("Modèle")
    lr, best = res.loc["Logistic Regression"], res["AUC-ROC"].idxmax()
    cmp_ = res.loc[list(dict.fromkeys(["Logistic Regression", best])), ["AUC-ROC", "PR-AUC", "Rappel", "Précision"]]
    st.dataframe(cmp_.round(3))
    gap = res.loc[best, "AUC-ROC"] - lr["AUC-ROC"]
    if gap < 0.03:
        st.success(f"L'écart d'AUC avec la régression logistique est faible ({gap:.3f}) : privilégiez-la, "
                   "ses coefficients sont directement lisibles et auditables.")
    else:
        st.info(f"{best} gagne {gap:.3f} point d'AUC sur la régression logistique : utilisez-le pour détecter, "
                "et SHAP pour expliquer chaque décision. Gardez la régression logistique comme référence de contrôle.")


# ======================================================================= navigation
PAGES = {"Tableau de bord": page_dashboard, "Analyse des départs": page_analysis,
         "Prédire un employé": page_predict, "Performance des modèles": page_models,
         "Facteurs de risque": page_factors, "Équité et fiabilité": page_fairness}

st.sidebar.title("👥 HR Analytics")
st.sidebar.caption("Prévention des départs d'employés")
page = st.sidebar.radio("Navigation", list(PAGES), label_visibility="collapsed")
st.sidebar.divider()
st.sidebar.caption(f"Modèle actif : **{INFO['model_name']}**")
PAGES[page]()