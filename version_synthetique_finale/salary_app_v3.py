"""
salary_app_v3.py — Outil d'Estimation Salariale — SNCF Réseau DRH
Conçu pour usage en entretien de recrutement : 30 secondes, 5 variables.
Youssef Riahi — EPSI Paris M2 — Stage SNCF Réseau 2025-2026
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import os

# ─── Config page ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Estimation Salariale — SNCF Réseau",
    page_icon="🚆",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ─── CSS personnalisé SNCF ────────────────────────────────────────────────
st.markdown("""
<style>
.stApp { background-color: #F8F6FF; }
.sncf-header {
    background: linear-gradient(135deg, #3B1F6E 0%, #7B4FD6 100%);
    padding: 28px 32px 20px 32px;
    border-radius: 16px;
    margin-bottom: 28px;
    box-shadow: 0 4px 20px rgba(59,31,110,0.25);
}
.sncf-header h1 { color: white !important; font-size: 1.8rem !important; margin: 0 0 4px 0 !important; font-weight: 700 !important; }
.sncf-header p { color: rgba(255,255,255,0.80) !important; font-size: 0.95rem !important; margin: 0 !important; }
.result-card { background: white; border: 2px solid #7B4FD6; border-radius: 16px; padding: 28px 32px; margin: 20px 0; box-shadow: 0 4px 24px rgba(123,79,214,0.15); }
.salary-main { font-size: 3.2rem; font-weight: 800; color: #3B1F6E; text-align: center; margin: 8px 0 4px 0; letter-spacing: -1px; }
.salary-sub { font-size: 1.1rem; color: #7B4FD6; text-align: center; margin-bottom: 16px; }
.salary-range { background: #F5F0FF; border-radius: 10px; padding: 12px 20px; text-align: center; font-size: 1.05rem; color: #3B1F6E; margin: 8px 0; }
.monthly { background: #E8F5E8; border-radius: 10px; padding: 10px 20px; text-align: center; font-size: 1.0rem; color: #1a7a1a; margin: 8px 0; }
.model-badge { background: #EBF3FF; border-radius: 8px; padding: 8px 16px; font-size: 0.82rem; color: #2c5aa0; margin-top: 12px; text-align: center; }
.ethic-box { background: #FFF8E1; border-left: 4px solid #F59E0B; border-radius: 0 8px 8px 0; padding: 10px 16px; font-size: 0.82rem; color: #92400E; margin-top: 16px; }
.stButton > button { background: linear-gradient(135deg, #3B1F6E, #7B4FD6) !important; color: white !important; border: none !important; border-radius: 12px !important; font-size: 1.1rem !important; font-weight: 600 !important; padding: 14px 28px !important; width: 100% !important; box-shadow: 0 4px 12px rgba(59,31,110,0.3) !important; }
label { font-weight: 600 !important; color: #3B1F6E !important; }
hr { border-color: #E8E0FF !important; margin: 24px 0 !important; }
</style>
""", unsafe_allow_html=True)

# ─── Fonction label diplôme (évite les f-string imbriquées) ───────────────
def get_diplome_label(d):
    if d == 'PhD':
        return 'PhD +30% vs Bachelor'
    elif d == "Master's":
        return "Master's +20% vs Bachelor"
    elif d == "Bachelor's":
        return "Bachelor's — référence marché"
    elif d == 'Baccalauréat':
        return 'Baccalauréat -12% vs Bachelor'
    else:
        return 'CAP/BEP -22% vs Bachelor'

# ─── Chargement du modèle ─────────────────────────────────────────────────
@st.cache_resource
def load_model():
    paths = [
        "models_v3/best_model.pkl",
        "models_v2/best_model.pkl",
        "best_model.pkl",
        "modele_final.pkl",
        "model.pkl",
    ]
    for p in paths:
        if os.path.exists(p):
            with open(p, "rb") as f:
                return pickle.load(f), p
    return None, None

@st.cache_data
def load_meta():
    paths = ["models_v3/best_meta.json", "models_v2/best_meta.json", "best_meta.json"]
    for p in paths:
        if os.path.exists(p):
            with open(p, encoding="utf-8") as f:
                return json.load(f)
    return {}

model, model_path = load_model()
meta = load_meta()

# ─── Données de référence ─────────────────────────────────────────────────
POSTES_FR = sorted([
    "Data Scientist", "Data Analyst", "Data Engineer", "ML Engineer",
    "Développeur Software", "Chef de Projet IT", "Architecte Cloud",
    "Consultant IT", "DevOps Engineer", "Ingénieur Systèmes",
    "Responsable RH", "Chargé RH", "Assistant Administratif", "Office Manager",
    "Contrôleur de Gestion", "Analyste Financier", "Comptable",
    "Directeur Financier", "Trader", "Consultant Finance", "Juriste",
    "Commercial", "Responsable Commercial", "Account Manager",
    "Business Developer", "Responsable Marketing", "Chargé de Communication",
    "Infirmier", "Médecin", "Pharmacien",
    "Enseignant", "Chercheur",
    "Technicien de Maintenance", "Logisticien", "Conducteur de Projet",
])

DIPLOMES = ["CAP/BEP", "Baccalauréat", "Bachelor's", "Master's", "PhD"]

# ─── EN-TÊTE ──────────────────────────────────────────────────────────────
st.markdown("""
<div class="sncf-header">
    <h1>🚆 Estimation Salariale — SNCF Réseau</h1>
    <p>Outil d'aide à la décision RH · Marché parisien · Usage en entretien</p>
</div>
""", unsafe_allow_html=True)

# ─── Message si modèle absent ─────────────────────────────────────────────
if model is None:
    st.error("Modèle non trouvé. Lancez d'abord notebook_M2_senior.ipynb pour entraîner le modèle.")
    st.info("Chemin attendu : models_v3/best_model.pkl")
    st.stop()

# ─── FORMULAIRE ───────────────────────────────────────────────────────────
st.markdown("### Profil du candidat")
st.markdown("*Renseignez les informations disponibles en entretien (30 secondes)*")

col1, col2 = st.columns(2)

with col1:
    poste = st.selectbox(
        "Intitulé de poste",
        POSTES_FR,
        index=POSTES_FR.index("Data Analyst") if "Data Analyst" in POSTES_FR else 0,
        help="Poste pour lequel le candidat postule"
    )
    diplome = st.selectbox(
        "Niveau de diplôme",
        DIPLOMES,
        index=DIPLOMES.index("Master's"),
        help="Dernier diplôme obtenu"
    )
    genre = st.selectbox(
        "Genre",
        ["Male", "Female"],
        format_func=lambda x: "Homme" if x == "Male" else "Femme",
        help="Genre du candidat"
    )

with col2:
    experience = st.slider(
        "Années d'expérience",
        min_value=0.0,
        max_value=35.0,
        value=5.0,
        step=0.5,
        help="Nombre total d'années d'expérience professionnelle"
    )
    age = st.slider(
        "Age",
        min_value=18,
        max_value=65,
        value=30,
        step=1,
        help="Age du candidat"
    )

    age_min_diplome = {
        "CAP/BEP": 18, "Baccalauréat": 18,
        "Bachelor's": 21, "Master's": 23, "PhD": 27
    }
    exp_max_possible = age - age_min_diplome.get(diplome, 21)
    if experience > exp_max_possible + 2:
        st.warning("Expérience elevée pour cet age.")
    else:
        st.success("Profil cohérent")

st.markdown("<hr>", unsafe_allow_html=True)

# ─── PRÉDICTION ───────────────────────────────────────────────────────────
predict_btn = st.button("Estimer le salaire du marché", use_container_width=True)

if predict_btn:
    input_data = pd.DataFrame([{
        "Age":                 age,
        "Gender":              genre,
        "Education Level":     diplome,
        "Job Title":           poste,
        "Years of Experience": experience,
    }])

    try:
        prediction = float(model.predict(input_data)[0])
        mae  = meta.get("MAE", 5000)
        r2   = meta.get("R2",  0.848)
        algo = meta.get("Algorithme", "CatBoost (Optuna)")
        rang = meta.get("Rang", 1)

        sal_low  = int(prediction - mae)
        sal_high = int(prediction + mae)
        sal_mois = int(prediction / 12)

        st.markdown(f"""
        <div class="result-card">
            <p style="text-align:center; color:#7B4FD6; font-weight:600; margin-bottom:4px; font-size:0.9rem;">
                ESTIMATION MARCHE PARISIEN 2024
            </p>
            <div class="salary-main">{int(prediction):,} EUR /an</div>
            <div class="salary-sub">Salaire net annuel estime — {poste}</div>
            <div class="salary-range">
                Fourchette de marche : <strong>{sal_low:,} EUR</strong> — <strong>{sal_high:,} EUR</strong> /an
                <br><small style="color:#888">Intervalle de confiance MAE ({int(mae):,} EUR)</small>
            </div>
            <div class="monthly">
                Mensuel estime : <strong>{sal_mois:,} EUR</strong> /mois net
            </div>
            <div class="model-badge">
                Modele : {algo} · R2={r2:.4f} · MAE={int(mae):,} EUR · Rang {rang}/12
                | Source : Dataset synthetique Paris 2024 calibre INSEE/APEC
            </div>
            <div class="ethic-box">
                Usage responsable : Cet outil est une aide a la decision basee sur des donnees de marche.
                La remuneration finale integre les grilles SNCF Reseau, la politique salariale interne
                et la negociation individuelle. Estimation fournie a titre indicatif uniquement.
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("#### Contexte marche parisien")
        c1, c2, c3 = st.columns(3)
        with c1:
            delta_marche = int((prediction - 32760) / 32760 * 100)
            st.metric("Mediane INSEE Paris", "32 760 EUR/an", delta=str(delta_marche) + "% vs marche")
        with c2:
            st.metric("Experience saisie", str(experience) + " ans", delta="+2.5%/an (APEC 2024)")
        with c3:
            st.metric("Diplome", diplome, delta="Impact significatif")

        with st.expander("Facteurs d'influence sur l'estimation"):
            facteurs = {
                "Metier (" + poste + ")":           "Variable la plus influente (SHAP #1)",
                "Diplome (" + diplome + ")":         get_diplome_label(diplome),
                "Experience (" + str(experience) + " ans)": "+" + str(round(experience * 2.5, 1)) + "% vs debutant (APEC 2024)",
                "Age (" + str(age) + " ans)":        "Correle a l'experience (r=0.89)",
                "Genre (" + genre + ")":             "Ecart marche observe H/F : +/-8% (INSEE 2023)",
            }
            for k, v in facteurs.items():
                st.markdown("- **" + k + "** : " + v)

        with st.expander("Postes similaires — Fourchettes indicatives"):
            ref_data = [
                {"Poste": "Data Analyst",      "Fourchette Paris net/an": "25 000 - 45 000 EUR", "Source": "APEC 2024"},
                {"Poste": "Data Scientist",    "Fourchette Paris net/an": "32 000 - 55 000 EUR", "Source": "APEC 2024"},
                {"Poste": "Data Engineer",     "Fourchette Paris net/an": "30 000 - 52 000 EUR", "Source": "APEC 2024"},
                {"Poste": "Chef de Projet IT", "Fourchette Paris net/an": "28 000 - 50 000 EUR", "Source": "APEC 2024"},
                {"Poste": "ML Engineer",       "Fourchette Paris net/an": "33 000 - 58 000 EUR", "Source": "APEC 2024"},
            ]
            st.dataframe(pd.DataFrame(ref_data), hide_index=True, use_container_width=True)

    except Exception as e:
        st.error("Erreur lors de la prediction : " + str(e))
        st.info("Verifiez que le modele a ete entraine sur les bonnes colonnes.")

# ─── PIED DE PAGE ─────────────────────────────────────────────────────────
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("""
<p style="text-align:center; color:#999; font-size:0.78rem;">
SNCF Reseau · Direction des Ressources Humaines · Service Relations Sociales<br>
Projet Data Science M2 — Youssef Riahi · EPSI Paris · Promo 2026<br>
Dataset synthetique calibre INSEE BTS 2023 + APEC Ile-de-France 2024 · 2 500 profils parisiens
</p>
""", unsafe_allow_html=True)
