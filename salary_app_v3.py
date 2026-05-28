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
/* Fond général */
.stApp { background-color: #F8F6FF; }

/* Header SNCF */
.sncf-header {
    background: linear-gradient(135deg, #3B1F6E 0%, #7B4FD6 100%);
    padding: 28px 32px 20px 32px;
    border-radius: 16px;
    margin-bottom: 28px;
    box-shadow: 0 4px 20px rgba(59,31,110,0.25);
}
.sncf-header h1 {
    color: white !important;
    font-size: 1.8rem !important;
    margin: 0 0 4px 0 !important;
    font-weight: 700 !important;
}
.sncf-header p {
    color: rgba(255,255,255,0.80) !important;
    font-size: 0.95rem !important;
    margin: 0 !important;
}

/* Carte résultat */
.result-card {
    background: white;
    border: 2px solid #7B4FD6;
    border-radius: 16px;
    padding: 28px 32px;
    margin: 20px 0;
    box-shadow: 0 4px 24px rgba(123,79,214,0.15);
}
.salary-main {
    font-size: 3.2rem;
    font-weight: 800;
    color: #3B1F6E;
    text-align: center;
    margin: 8px 0 4px 0;
    letter-spacing: -1px;
}
.salary-sub {
    font-size: 1.1rem;
    color: #7B4FD6;
    text-align: center;
    margin-bottom: 16px;
}
.salary-range {
    background: #F5F0FF;
    border-radius: 10px;
    padding: 12px 20px;
    text-align: center;
    font-size: 1.05rem;
    color: #3B1F6E;
    margin: 8px 0;
}
.monthly {
    background: #E8F5E8;
    border-radius: 10px;
    padding: 10px 20px;
    text-align: center;
    font-size: 1.0rem;
    color: #1a7a1a;
    margin: 8px 0;
}

/* Bandeau modèle */
.model-badge {
    background: #EBF3FF;
    border-radius: 8px;
    padding: 8px 16px;
    font-size: 0.82rem;
    color: #2c5aa0;
    margin-top: 12px;
    text-align: center;
}

/* Avertissement éthique */
.ethic-box {
    background: #FFF8E1;
    border-left: 4px solid #F59E0B;
    border-radius: 0 8px 8px 0;
    padding: 10px 16px;
    font-size: 0.82rem;
    color: #92400E;
    margin-top: 16px;
}

/* Bouton principal */
.stButton > button {
    background: linear-gradient(135deg, #3B1F6E, #7B4FD6) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    font-size: 1.1rem !important;
    font-weight: 600 !important;
    padding: 14px 28px !important;
    width: 100% !important;
    transition: all 0.2s !important;
    box-shadow: 0 4px 12px rgba(59,31,110,0.3) !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(59,31,110,0.4) !important;
}

/* Labels formulaire */
label { font-weight: 600 !important; color: #3B1F6E !important; }

/* Séparateur */
hr { border-color: #E8E0FF !important; margin: 24px 0 !important; }
</style>
""", unsafe_allow_html=True)

# ─── Chargement du modèle ─────────────────────────────────────────────────
@st.cache_resource
def load_model():
    paths = [
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
    paths = ["models_v2/best_meta.json", "best_meta.json"]
    for p in paths:
        if os.path.exists(p):
            with open(p, encoding="utf-8") as f:
                return json.load(f)
    return {}

model, model_path = load_model()
meta = load_meta()

# ─── Données de référence ─────────────────────────────────────────────────
# 35 métiers calibrés marché parisien
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
    <p>Outil d'aide à la décision RH · Marché parisien 2024 · Usage en entretien</p>
</div>
""", unsafe_allow_html=True)

# ─── Message si modèle absent ─────────────────────────────────────────────
if model is None:
    st.error("⚠️ Modèle non trouvé. Lancez d'abord `notebook_M2_senior.ipynb` pour entraîner le modèle.")
    st.info("Chemin attendu : `models_v2/best_model.pkl`")
    st.stop()

# ─── FORMULAIRE ───────────────────────────────────────────────────────────
st.markdown("### 📋 Profil du candidat")
st.markdown("*Renseignez les informations disponibles en entretien (30 secondes)*")

col1, col2 = st.columns(2)

with col1:
    poste = st.selectbox(
        "🎯 Intitulé de poste",
        POSTES_FR,
        index=POSTES_FR.index("Data Analyst") if "Data Analyst" in POSTES_FR else 0,
        help="Poste pour lequel le candidat postule"
    )
    diplome = st.selectbox(
        "🎓 Niveau de diplôme",
        DIPLOMES,
        index=DIPLOMES.index("Master's"),
        help="Dernier diplôme obtenu"
    )
    genre = st.selectbox(
        "👤 Genre",
        ["Male", "Female"],
        format_func=lambda x: "Homme" if x == "Male" else "Femme",
        help="Genre du candidat"
    )

with col2:
    experience = st.slider(
        "📅 Années d'expérience",
        min_value=0.0,
        max_value=35.0,
        value=5.0,
        step=0.5,
        help="Nombre total d'années d'expérience professionnelle"
    )
    age = st.slider(
        "🎂 Âge",
        min_value=18,
        max_value=65,
        value=30,
        step=1,
        help="Âge du candidat"
    )

    # Indicateur de cohérence âge/expérience
    age_min_diplome = {"CAP/BEP": 18, "Baccalauréat": 18,
                       "Bachelor's": 21, "Master's": 23, "PhD": 27}
    exp_max_possible = age - age_min_diplome.get(diplome, 21)
    if experience > exp_max_possible + 2:
        st.warning(f"⚠️ {experience:.0f} ans d'expérience pour {age} ans semble élevé.")
    elif experience <= exp_max_possible:
        st.success(f"✅ Profil cohérent ({experience:.0f} ans exp. / {age} ans)")

st.markdown("<hr>", unsafe_allow_html=True)

# ─── PRÉDICTION ───────────────────────────────────────────────────────────
predict_btn = st.button("💰 Estimer le salaire du marché", use_container_width=True)

if predict_btn:
    # Construire le DataFrame d'entrée
    input_data = pd.DataFrame([{
        "Age":                 age,
        "Gender":              genre,
        "Education Level":     diplome,
        "Job Title":           poste,
        "Years of Experience": experience,
    }])

    try:
        prediction = float(model.predict(input_data)[0])
        mae = meta.get("MAE", 5000)
        r2  = meta.get("R2",  0.848)
        algo = meta.get("Algorithme", "CatBoost (Optuna)")
        rang = meta.get("Rang", 1)

        sal_low  = int(prediction - mae)
        sal_high = int(prediction + mae)
        sal_mois = int(prediction / 12)

        # ── Carte résultat ──────────────────────────────────────────────
        st.markdown(f"""
        <div class="result-card">
            <p style="text-align:center; color:#7B4FD6; font-weight:600; margin-bottom:4px; font-size:0.9rem;">
                ESTIMATION MARCHÉ PARISIEN 2024
            </p>
            <div class="salary-main">{int(prediction):,} €<span style="font-size:1.4rem; font-weight:400"> /an</span></div>
            <div class="salary-sub">Salaire net annuel estimé — {poste}</div>
            <div class="salary-range">
                📊 Fourchette de marché : <strong>{sal_low:,} €</strong> — <strong>{sal_high:,} €</strong> /an
                <br><small style="color:#888">Intervalle de confiance ±MAE ({int(mae):,} €)</small>
            </div>
            <div class="monthly">
                💳 Mensuel estimé : <strong>{sal_mois:,} €</strong> /mois net
            </div>
            <div class="model-badge">
                🤖 Modèle : {algo} · R²={r2:.4f} · MAE={int(mae):,}€ · Rang {rang}/12
                &nbsp;|&nbsp; Source : Dataset synthétique Paris 2024 calibré INSEE/APEC
            </div>
            <div class="ethic-box">
                ⚖️ <strong>Usage responsable :</strong> Cet outil est une aide à la décision basée sur des données de marché.
                La rémunération finale intègre les grilles SNCF Réseau, la politique salariale interne et la négociation individuelle.
                L'estimation est fournie à titre indicatif uniquement.
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Contexte marché ──────────────────────────────────────────────
        st.markdown("#### 📈 Contexte marché parisien")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Médiane INSEE Paris",  "32 760 €/an",
                      delta=f"{int((prediction-32760)/32760*100):+d}% vs marché")
        with c2:
            st.metric("Expérience saisie",   f"{experience:.1f} ans",
                      delta=f"+2.5%/an (APEC 2024)")
        with c3:
            st.metric("Diplôme",             diplome,
                      delta="Impact significatif sur salaire")

        # ── Facteurs d'influence ─────────────────────────────────────────
        with st.expander("🔍 Facteurs d'influence sur l'estimation"):
            facteurs = {
                f"📌 Métier ({poste})":           "Variable la plus influente (SHAP #1)",
                f"🎓 Diplôme ({diplome})":         f"{'PhD +30%' if diplome=='PhD' else 'Master +20%' if diplome==\"Master's\" else 'Bachelor référence' if diplome==\"Bachelor's\" else 'Bac -12%' if diplome=='Baccalauréat' else 'CAP/BEP -22%'}  vs Bachelor's",
                f"📅 Expérience ({experience} ans)": f"+{experience*2.5:.1f}% vs débutant (APEC 2024)",
                f"🎂 Âge ({age} ans)":              "Corrélé à l'expérience (r=0.89)",
                f"👤 Genre ({genre})":              "Écart marché observé H/F : +/-8% (INSEE 2023)",
            }
            for k, v in facteurs.items():
                st.markdown(f"- **{k}** → {v}")

        # ── Comparaison postes similaires ────────────────────────────────
        with st.expander("🏢 Postes similaires — Fourchettes indicatives"):
            postes_proches = {
                "Data Analyst":      "25 000 – 45 000 €",
                "Data Scientist":    "32 000 – 55 000 €",
                "Data Engineer":     (30000, 52000),
                "Chef de Projet IT": (28000, 50000),
                "ML Engineer":       (33000, 58000),
            }
            ref_data = []
            for p_name, p_range in postes_proches.items():
                if isinstance(p_range, tuple):
                    p_str = f"{p_range[0]:,} – {p_range[1]:,} €"
                else:
                    p_str = p_range
                ref_data.append({
                    "Poste": p_name,
                    "Fourchette marché Paris (net/an)": p_str,
                    "Source": "APEC 2024"
                })
            st.dataframe(pd.DataFrame(ref_data), hide_index=True, use_container_width=True)

    except Exception as e:
        st.error(f"Erreur lors de la prédiction : {e}")
        st.info("Vérifiez que le modèle a été entraîné sur les bonnes colonnes.")

# ─── PIED DE PAGE ─────────────────────────────────────────────────────────
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("""
<p style="text-align:center; color:#999; font-size:0.78rem;">
SNCF Réseau · Direction des Ressources Humaines · Service Relations Sociales<br>
Projet Data Science M2 — Youssef Riahi · EPSI Paris · Promo 2026<br>
Dataset : Synthétique calibré INSEE BTS 2023 + APEC Île-de-France 2024 · 2 500 profils parisiens
</p>
""", unsafe_allow_html=True)
