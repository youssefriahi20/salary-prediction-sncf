"""
📊 DASHBOARD ANALYTIQUE — Projet Prédiction Salariale Marché Parisien
=====================================================================
Tableau de bord interactif pour analyser :
- Les performances des 12 modèles entraînés
- La distribution des salaires du marché parisien
- L'impact des variables sur la prédiction (via SHAP)
- Les KPIs business clés

Lancement : streamlit run dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle
import json
import os

# ═════════════════════════════════════════════════════════════
# CONFIGURATION DE LA PAGE
# ═════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Dashboard — Prédiction Salariale Paris",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Style CSS personnalisé
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1F3864 0%, #2E5090 100%);
        padding: 1.5rem 2rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #C8A96E;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .stMetric { background-color: #F8F9FA; padding: 1rem; border-radius: 8px; }
    h1, h2, h3 { color: #1F3864; }
</style>
""", unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════
# HEADER
# ═════════════════════════════════════════════════════════════
st.markdown("""
<div class="main-header">
    <h1 style="color:white; margin:0;">📊 Dashboard Analytique</h1>
    <p style="color:#C8A96E; font-size:1.1rem; margin:0.5rem 0 0 0;">
        Projet Data Science M2 — Prédiction Salariale Marché Parisien
    </p>
</div>
""", unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════
# CHARGEMENT DES DONNÉES
# ═════════════════════════════════════════════════════════════
@st.cache_data
def load_data():
    df = pd.read_csv('dataset_paris_synthetique_2024.csv')
    if 'Salary_USD_original' in df.columns:
        df = df.drop(columns=['Salary_USD_original'])
    return df

@st.cache_data
def load_results():
    paths = ['models_v2/results_final.csv', 'results_final.csv']
    for p in paths:
        if os.path.exists(p):
            return pd.read_csv(p)
    return None

@st.cache_resource
def load_model():
    paths = ['models_v2/best_model.pkl', 'best_model.pkl']
    for p in paths:
        if os.path.exists(p):
            with open(p, 'rb') as f:
                return pickle.load(f)
    return None

@st.cache_data
def load_meta():
    paths = ['models_v2/best_meta.json', 'best_meta.json']
    for p in paths:
        if os.path.exists(p):
            with open(p, 'r', encoding='utf-8') as f:
                return json.load(f)
    return None

df = load_data()
df_results = load_results()
meta = load_meta()
best_model = load_model()

# ═════════════════════════════════════════════════════════════
# SIDEBAR — FILTRES
# ═════════════════════════════════════════════════════════════
st.sidebar.markdown("## 🎛️ Filtres")
genres_filter = st.sidebar.multiselect(
    "Genre", options=df['Gender'].unique().tolist(),
    default=df['Gender'].unique().tolist()
)
edu_filter = st.sidebar.multiselect(
    "Niveau d'études", options=df['Education Level'].unique().tolist(),
    default=df['Education Level'].unique().tolist()
)
age_range = st.sidebar.slider(
    "Tranche d'âge", int(df['Age'].min()), int(df['Age'].max()),
    (int(df['Age'].min()), int(df['Age'].max()))
)

df_filtered = df[
    (df['Gender'].isin(genres_filter)) &
    (df['Education Level'].isin(edu_filter)) &
    (df['Age'].between(age_range[0], age_range[1]))
]

st.sidebar.markdown(f"**📌 {len(df_filtered)} profils** correspondent")
st.sidebar.markdown("---")
st.sidebar.markdown("### ℹ️ À propos")
st.sidebar.info(
    "Dashboard généré dans le cadre du projet M2 EPSI Paris — "
    "Stage SNCF Réseau. Données : 2500 profils marché parisien."
)

# ═════════════════════════════════════════════════════════════
# KPIs PRINCIPAUX (4 metric cards)
# ═════════════════════════════════════════════════════════════
st.markdown("## 🎯 Indicateurs clés")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric(
        "Profils analysés",
        f"{len(df_filtered):,}",
        delta=f"sur {len(df):,} au total"
    )
with col2:
    st.metric(
        "Salaire médian",
        f"{int(df_filtered['Salary'].median()):,} €",
        delta=f"σ = {int(df_filtered['Salary'].std()):,} €"
    )
with col3:
    if meta:
        st.metric(
            "R² du meilleur modèle",
            f"{meta['r2']:.4f}",
            delta=f"sur {meta.get('rank_total', 12)} modèles testés"
        )
with col4:
    if meta:
        st.metric(
            "MAE",
            f"{int(meta['mae']):,} €",
            delta=f"MAPE = {meta['mape']:.1f}%"
        )

st.markdown("---")

# ═════════════════════════════════════════════════════════════
# ROW 1 : Distribution des salaires + Comparaison modèles
# ═════════════════════════════════════════════════════════════
col_g, col_d = st.columns([1, 1])

with col_g:
    st.markdown("### 💰 Distribution des salaires")
    fig_hist = px.histogram(
        df_filtered, x='Salary', nbins=40,
        color_discrete_sequence=['#1F3864'],
        labels={'Salary': 'Salaire annuel brut (€)', 'count': 'Nombre de profils'},
    )
    fig_hist.add_vline(
        x=df_filtered['Salary'].median(),
        line_dash="dash", line_color="#C8A96E",
        annotation_text=f"Médiane : {int(df_filtered['Salary'].median()):,} €",
        annotation_position="top right",
    )
    fig_hist.update_layout(height=400, showlegend=False)
    st.plotly_chart(fig_hist, use_container_width=True)

with col_d:
    st.markdown("### 🏆 Comparaison des 12 modèles")
    if df_results is not None:
        df_plot = df_results.sort_values('R²', ascending=True).copy()
        # Couleurs selon catégorie
        def get_color(name):
            if 'Régression' in name or 'Ridge' in name:
                return '#C44E52'
            elif 'default' in name:
                return '#E8A87C'
            elif 'Optuna' in name:
                return '#4C72B0'
            elif 'Stacking' in name or '🏆' in name:
                return '#55A868'
            return '#8172B2'

        df_plot['color'] = df_plot['Modèle'].apply(get_color)
        fig_bar = go.Figure(go.Bar(
            x=df_plot['R²'],
            y=df_plot['Modèle'],
            orientation='h',
            marker=dict(color=df_plot['color']),
            text=[f"{r:.4f}" for r in df_plot['R²']],
            textposition='outside',
        ))
        fig_bar.add_vline(x=0.85, line_dash="dash", line_color="#C44E52",
                            annotation_text="Cible R² ≥ 0.85", annotation_position="top")
        fig_bar.update_layout(
            xaxis_title="R² (Coefficient de détermination)",
            height=400, margin=dict(l=10, r=10, t=10, b=10),
            xaxis=dict(range=[0.74, 0.88])
        )
        st.plotly_chart(fig_bar, use_container_width=True)

# ═════════════════════════════════════════════════════════════
# ROW 2 : Analyse par variables
# ═════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("## 📊 Analyse par profil")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### Par niveau d'études")
    fig = px.box(
        df_filtered, x='Education Level', y='Salary',
        color='Education Level',
        category_orders={"Education Level": ["Bachelor's", "Master's", "PhD"]},
        color_discrete_sequence=['#E8A87C', '#4C72B0', '#55A868'],
    )
    fig.update_layout(height=350, showlegend=False, xaxis_title='', yaxis_title='Salaire (€)')
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.markdown("### Par genre")
    fig = px.box(
        df_filtered, x='Gender', y='Salary', color='Gender',
        color_discrete_sequence=['#4C72B0', '#C44E52'],
    )
    fig.update_layout(height=350, showlegend=False, xaxis_title='', yaxis_title='Salaire (€)')
    st.plotly_chart(fig, use_container_width=True)

with col3:
    st.markdown("### Par expérience")
    fig = px.scatter(
        df_filtered, x='Years of Experience', y='Salary',
        color='Education Level',
        category_orders={"Education Level": ["Bachelor's", "Master's", "PhD"]},
        color_discrete_sequence=['#E8A87C', '#4C72B0', '#55A868'],
        opacity=0.6,
    )
    fig.update_layout(height=350, legend=dict(orientation="h", y=-0.2))
    st.plotly_chart(fig, use_container_width=True)

# ═════════════════════════════════════════════════════════════
# ROW 3 : Top postes les mieux payés
# ═════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("## 💼 Top 15 postes les mieux rémunérés (marché parisien)")

top_jobs = (df_filtered.groupby('Job Title')['Salary']
              .agg(['mean', 'count'])
              .query('count >= 3')  # min 3 obs pour fiabilité
              .sort_values('mean', ascending=False)
              .head(15)
              .reset_index())
top_jobs.columns = ['Job Title', 'Salaire moyen (€)', 'Nb profils']

fig_top = px.bar(
    top_jobs.sort_values('Salaire moyen (€)', ascending=True),
    x='Salaire moyen (€)', y='Job Title',
    orientation='h',
    color='Salaire moyen (€)',
    color_continuous_scale='Viridis',
    text='Nb profils',
)
fig_top.update_traces(texttemplate='n=%{text}', textposition='inside')
fig_top.update_layout(height=500, yaxis_title='', coloraxis_showscale=False)
st.plotly_chart(fig_top, use_container_width=True)

# ═════════════════════════════════════════════════════════════
# ROW 4 : Tableau interactif des modèles
# ═════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("## 📋 Tableau détaillé des 12 modèles")

if df_results is not None:
    # Tri par défaut sur R²
    df_show = df_results.sort_values('R²', ascending=False).reset_index(drop=True)
    df_show['Rang'] = range(1, len(df_show) + 1)
    df_show = df_show[['Rang', 'Modèle', 'R²', 'MAE', 'RMSE', 'MAPE']]
    st.dataframe(
        df_show.style
            .background_gradient(subset=['R²'], cmap='RdYlGn')
            .background_gradient(subset=['MAE', 'RMSE', 'MAPE'], cmap='RdYlGn_r')
            .format({'R²': '{:.4f}', 'MAE': '{:,.0f} €',
                      'RMSE': '{:,.0f} €', 'MAPE': '{:.2f} %'}),
        use_container_width=True,
        hide_index=True,
    )

# ═════════════════════════════════════════════════════════════
# FOOTER
# ═════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray; font-size: 0.9rem;'>
        🎓 Projet M2 EPSI Paris — MSc Chef de Projet Expert en IA<br>
        Stage SNCF Réseau — Direction des Ressources Humaines — Année 2025/2026<br>
        Dataset synthétique de 2 500 profils RH calibré sur les statistiques INSEE 2023 et APEC Île-de-France 2024
    </div>
    """,
    unsafe_allow_html=True,
)
