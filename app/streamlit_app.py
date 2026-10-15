import os
import requests
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Attrition RH – XAI", layout="wide")
sns.set_theme(style="whitegrid")


def get_api_url():
    try:
        return st.secrets["API_URL"]
    except Exception:
        return os.getenv("API_URL", "http://127.0.0.1:8000")


API_URL = get_api_url().rstrip("/")


@st.cache_data(ttl=3600, show_spinner="Appel de l'API…")
def api_get(path: str):
    r = requests.get(f"{API_URL}{path}", timeout=90)
    r.raise_for_status()
    return r.json()


def api_post(path: str, payload: dict):
    r = requests.post(f"{API_URL}{path}", json=payload, timeout=90)
    if r.status_code != 200:
        raise RuntimeError(r.json().get("detail", r.text))
    return r.json()


def plot_contribs(items, title):
    d = pd.DataFrame(items).sort_values("contribution")
    colors = ["#D1495B" if v > 0 else "#4C9F70" for v in d["contribution"]]
    fig, ax = plt.subplots(figsize=(6.5, 0.45 * len(d) + 1.3))
    ax.barh(d["label"], d["contribution"], color=colors)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_title(title)
    ax.set_xlabel("Contribution (rouge = augmente le risque)")
    plt.tight_layout()
    return fig


# ------------------------------------------------------------------ pages 1 à 4
def page_donnees():
    st.header("Modules 1 & 2 – Données et préparation")
    st.info("À compléter : cellules 1 à 4 du notebook (chargement, EDA, préparation).")


def page_modeles():
    st.header("Modules 3 & 4 – Modélisation et évaluation")
    st.info("À compléter : cellules 5 à 8 du notebook (modèles, comparaison, ROC, seuil).")


def page_xai():
    st.header("Module 5 – Explainable AI")
    st.info("À brancher plus tard sur les endpoints /explain/* de l'API (déjà prêts).")


# ------------------------------------------------------------------ page 6
def page_api():
    st.header("Module 6 – API d'interprétation (FastAPI)")
    try:
        info = api_get("/model/info")
        meta = api_get("/features")
    except Exception as e:
        st.error(f"API injoignable ({API_URL}). Si elle est en veille, patiente 1 minute et recharge.\n\n{e}")
        return

    st.caption(f"API : {API_URL} · [Documentation Swagger]({API_URL}/docs)")
    c = st.columns(5)
    c[0].metric("Modèle", info["model_name"])
    c[1].metric("AUC-ROC", info["metrics"]["AUC-ROC"])
    c[2].metric("Rappel", info["metrics"]["Rappel"])
    c[3].metric("Précision", info["metrics"]["Précision"])
    c[4].metric("Seuil par défaut", info["threshold"])

    # ----- Explications globales
    st.subheader("Explications globales")
    imp = pd.DataFrame(api_get("/explain/global?top_k=15")["importance"])
    g1, g2 = st.columns(2)
    for ax_col, col, ttl in [(g1, "shap_mean_abs", "Importance SHAP (moyenne |φ|)"),
                             (g2, "permutation", "Importance par permutation (baisse d'AUC)")]:
        fig, ax = plt.subplots(figsize=(6, 5))
        d = imp.sort_values(col, ascending=False)
        sns.barplot(data=d, x=col, y="feature", color="#4C78A8", ax=ax)
        ax.set_title(ttl); ax.set_ylabel("")
        plt.tight_layout()
        ax_col.pyplot(fig)

    # ----- Explication locale
    st.subheader("Explication d'un individu")
    src = st.radio("Source", ["Individu fictif", "Individu du jeu de test"], horizontal=True)
    defaults = {f["name"]: f["default"] for f in meta}
    idx, actual = 0, None
    if src == "Individu du jeu de test":
        idx = st.number_input("Index dans le jeu de test", 0, info["n_test_samples"] - 1, 0)
        s = api_get(f"/samples/{int(idx)}")
        defaults.update(s["features"])
        actual = s["actual"]
        st.write("Réalité : " + ("**l'employé est parti**" if actual else "**l'employé est resté**"))

    tag = f"{src}_{idx}"
    values = {}
    with st.expander("Caractéristiques de l'employé", expanded=True):
        cols = st.columns(3)
        for i, f in enumerate(meta):
            with cols[i % 3]:
                key, name = f"{f['name']}_{tag}", f["name"]
                if f["type"] == "categorical":
                    opts = f["options"]
                    values[name] = st.selectbox(name, opts, index=opts.index(defaults[name]), key=key)
                elif f.get("integer"):
                    values[name] = st.number_input(name, int(f["min"]), int(f["max"]),
                                                   int(defaults[name]), step=1, key=key)
                else:
                    values[name] = st.number_input(name, f["min"], f["max"],
                                                   float(defaults[name]), key=key)

    thr = st.slider("Seuil de décision", 0.05, 0.95, float(info["threshold"]), 0.01)
    top_k = st.slider("Nombre de variables affichées", 3, 15, 8)

    if st.button("Analyser", type="primary"):
        try:
            st.session_state["res"] = api_post("/explain/compare",
                                               {"features": values, "threshold": thr, "top_k": top_k})
        except Exception as e:
            st.error(f"Erreur API : {e}")

    res = st.session_state.get("res")
    if res:
        m = st.columns(3)
        m[0].metric("Probabilité de départ", f"{res['probability']:.1%}")
        m[1].metric("Décision", res["decision_label"])
        m[2].metric("Niveau de risque", res["risk_level"])
        st.success(res["summary"])

        a, b = st.columns(2)
        a.pyplot(plot_contribs(res["shap"], "SHAP"))
        b.pyplot(plot_contribs(res["lime"], "LIME"))
        st.write(f"**Accord SHAP/LIME** : {res['overlap_ratio']:.0%} de variables en commun, "
                 f"{res['same_sign_ratio']:.0%} de même sens.")


PAGES = {"1-2 · Données & préparation": page_donnees,
         "3-4 · Modèles & évaluation": page_modeles,
         "5 · Explainable AI": page_xai,
         "6 · API d'interprétation": page_api}

st.sidebar.title("Attrition RH – XAI")
PAGES[st.sidebar.radio("Navigation", list(PAGES))]()