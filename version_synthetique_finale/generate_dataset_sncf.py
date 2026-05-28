"""
Dataset Synthétique — Marché Parisien 2024
5 variables RH : Age, Gender, Education Level, Job Title, Years of Experience, Salary
Calibré INSEE BTS 2023 + APEC Île-de-France 2024
random_state = 42
"""
import numpy as np
import pandas as pd

RANDOM_STATE = 42
N = 2500
np.random.seed(RANDOM_STATE)

METIERS = {
    # Tech & Data
    "Data Scientist":           (36000, 8000,  "Master's",   0.040),
    "Data Analyst":             (28000, 6000,  "Master's",   0.050),
    "Data Engineer":            (35000, 8000,  "Master's",   0.040),
    "ML Engineer":              (38000, 9000,  "Master's",   0.030),
    "Développeur Software":     (30000, 7000,  "Master's",   0.060),
    "Chef de Projet IT":        (34000, 8000,  "Master's",   0.040),
    "Architecte Cloud":         (42000, 9000,  "Master's",   0.020),
    "Consultant IT":            (35000, 8000,  "Master's",   0.030),
    "DevOps Engineer":          (36000, 8000,  "Master's",   0.025),
    "Ingénieur Systèmes":       (32000, 7000,  "Master's",   0.025),
    # RH & Administration
    "Responsable RH":           (30000, 6000,  "Master's",   0.030),
    "Chargé RH":                (22000, 4000,  "Bachelor's", 0.040),
    "Assistant Administratif":  (20000, 3000,  "Bachelor's", 0.050),
    "Office Manager":           (24000, 4000,  "Bachelor's", 0.025),
    # Finance & Gestion
    "Contrôleur de Gestion":    (32000, 7000,  "Master's",   0.030),
    "Analyste Financier":       (35000, 8000,  "Master's",   0.030),
    "Comptable":                (23000, 4000,  "Bachelor's", 0.040),
    "Directeur Financier":      (55000, 14000, "Master's",   0.015),
    "Trader":                   (48000, 16000, "Master's",   0.010),
    "Consultant Finance":       (38000, 9000,  "Master's",   0.020),
    "Juriste":                  (32000, 7000,  "Master's",   0.020),
    # Commercial & Marketing
    "Commercial":               (24000, 6000,  "Bachelor's", 0.050),
    "Responsable Commercial":   (37000, 9000,  "Master's",   0.030),
    "Account Manager":          (29000, 6000,  "Bachelor's", 0.030),
    "Business Developer":       (28000, 6000,  "Bachelor's", 0.025),
    "Responsable Marketing":    (30000, 7000,  "Master's",   0.025),
    "Chargé de Communication":  (22000, 4000,  "Bachelor's", 0.025),
    # Santé
    "Infirmier":                (22000, 3500,  "Bachelor's", 0.030),
    "Médecin":                  (50000, 14000, "PhD",        0.010),
    "Pharmacien":               (30000, 5000,  "PhD",        0.010),
    # Enseignement & Recherche
    "Enseignant":               (20000, 3500,  "Master's",   0.030),
    "Chercheur":                (27000, 5000,  "PhD",        0.020),
    # Services & Logistique
    "Technicien de Maintenance":(20000, 3500,  "Bachelor's", 0.030),
    "Logisticien":              (21000, 3500,  "Bachelor's", 0.025),
    "Conducteur de Projet":     (27000, 5500,  "Bachelor's", 0.020),
}

PRIME_GENRE   = {"Male": 0.08, "Female": -0.08}
PRIME_DIPLOME = {
    "CAP/BEP":      -0.22,
    "Baccalauréat": -0.12,
    "Bachelor's":    0.00,
    "Master's":      0.20,
    "PhD":           0.30,
}

metiers_list  = list(METIERS.keys())
poids_metiers = np.array([METIERS[m][3] for m in metiers_list])
poids_metiers /= poids_metiers.sum()

records = []
for _ in range(N):
    metier = np.random.choice(metiers_list, p=poids_metiers)
    sal_base, sal_std, dip_typique, _ = METIERS[metier]

    # Diplôme corrélé au métier
    if dip_typique == "PhD":
        dp = [0.01, 0.02, 0.07, 0.20, 0.70]
    elif dip_typique == "Master's":
        dp = [0.03, 0.06, 0.18, 0.55, 0.18]
    else:
        dp = [0.10, 0.18, 0.52, 0.17, 0.03]
    diplome = np.random.choice(
        ["CAP/BEP","Baccalauréat","Bachelor's","Master's","PhD"], p=dp)

    # Âge plancher selon diplôme
    age_floor = {"CAP/BEP":18,"Baccalauréat":18,
                 "Bachelor's":21,"Master's":23,"PhD":27}[diplome]
    age = int(np.clip(np.random.normal(38, 9), age_floor, 62))

    # Expérience corrélée à l'âge
    exp_max  = max(age - age_floor, 0)
    exp_mean = exp_max * 0.72
    exp = round(float(np.clip(
        np.random.normal(exp_mean, max(exp_max*0.22, 0.5)), 0, exp_max)), 1)

    genre = np.random.choice(["Male","Female"], p=[0.53, 0.47])

    # Salaire log-normal
    mu    = np.log(sal_base) - 0.5*(0.20**2)
    sal   = np.random.lognormal(mu, 0.20)
    sal  *= (1 + 0.025 * exp)
    sal  *= (1 + PRIME_DIPLOME[diplome])
    sal  *= (1 + PRIME_GENRE[genre])
    sal  *= np.random.normal(1.0, 0.04)   # bruit σ=4%
    sal   = int(np.clip(round(sal/100)*100, 17764, 200000))

    records.append({
        "Age":                 age,
        "Gender":              genre,
        "Education Level":     diplome,
        "Job Title":           metier,
        "Years of Experience": exp,
        "Salary":              sal,
    })

df = pd.DataFrame(records)

# Validation rapide
print("=" * 60)
print("DATASET SYNTHÉTIQUE PARIS 2024 — 6 VARIABLES RH")
print("=" * 60)
print(f"Lignes      : {len(df):,}")
print(f"Médiane     : {int(df['Salary'].median()):,} € (ref INSEE: 32 760 €)")
print(f"Moyenne     : {int(df['Salary'].mean()):,} €")
print(f"Min / Max   : {int(df['Salary'].min()):,} € / {int(df['Salary'].max()):,} €")
print(f"Métiers     : {df['Job Title'].nunique()}")
print(f"\nCorr. Expérience→Salaire : {df['Years of Experience'].corr(df['Salary']):.3f}")
print(f"Corr. Age→Salaire        : {df['Age'].corr(df['Salary']):.3f}")

df.to_csv("dataset_paris_synthetique_2024.csv", index=False)
print(f"\n✅ Sauvegardé : dataset_paris_synthetique_2024.csv")
print(f"   Colonnes : {list(df.columns)}")
