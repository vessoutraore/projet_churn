# streamlit/app.py

import pickle
from pathlib import Path

import numpy as np
import streamlit as st

# =====================================================
# 0. CONFIG GLOBALE & CHARGEMENT DU MODELE
# =====================================================

st.set_page_config(
    page_title="Churn Télécom – Guinée",
    page_icon="📡",
    layout="wide",
)

# Chemins
BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "model" / "model.pkl"
IMG_DIR = BASE_DIR / "img"

@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

model = load_model()

# Petits dictionnaires d’encodage (à adapter si besoin
# pour coller exactement à ce que tu as utilisé dans le notebook)
REGION_MAP = {
    "Conakry": 0,
    "Kankan": 1,
    "Labé": 2,
    "N’Zérékoré": 3,
    "Boké": 4,
    "Faranah": 5,
}
SEXE_MAP = {"M": 0, "F": 1}
ABO_MAP = {"Prépayé": 0, "Postpayé": 1}
OUI_NON_MAP = {"Non": 0, "Oui": 1}


# =====================================================
# 1. UN PEU DE STYLE
# =====================================================

CUSTOM_CSS = """
<style>
/* fond général */
[data-testid="stAppViewContainer"] {
    background: #f7f8fc;
}

/* bloc principal */
.main-block {
    padding: 1.5rem 2rem;
    background-color: #ffffff;
    border-radius: 18px;
    box-shadow: 0 8px 25px rgba(15, 23, 42, 0.08);
}

/* titres */
h1, h2, h3 {
    font-family: "Segoe UI", system-ui, -apple-system, BlinkMacSystemFont;
}

/* bouton prédiction */
.stButton>button {
    border-radius: 999px;
    background: linear-gradient(90deg, #2563eb, #0ea5e9);
    color: white;
    border: none;
    padding: 0.6rem 1.6rem;
    font-weight: 600;
}

/* petites cartes métriques */
.metric-card {
    padding: 1rem 1.5rem;
    border-radius: 16px;
    background: #0f172a;
    color: #f9fafb;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# =====================================================
# 2. HEADER
# =====================================================

col_logo, col_title = st.columns([1, 3])

with col_logo:
    logo_path = IMG_DIR / "logo_app.jpg"
    if logo_path.exists():
        st.image(str(logo_path), use_container_width=True)
    else:
        st.write("📡")

with col_title:
    st.markdown("### Churn Télécom – Guinée")
    st.markdown(
        "Démo basée sur un **modèle Random Forest** entraîné sur des données "
        "simulées de clients télécom en Guinée."
    )

st.markdown("---")

tab_predict, tab_model = st.tabs(["🔮 Prédiction client", "📊 À propos du modèle"])


# =====================================================
# 3. ONGLET PREDICTION
# =====================================================

with tab_predict:
    st.markdown('<div class="main-block">', unsafe_allow_html=True)

    st.subheader("🧑‍💼 Informations client")

    c1, c2, c3 = st.columns(3)

    with c1:
        region = st.selectbox("Région", list(REGION_MAP.keys()))
        sexe = st.selectbox("Sexe", ["M", "F"])
        age = st.slider("Âge", 18, 80, 32)

    with c2:
        revenu = st.number_input(
            "Revenu estimé (GNF)",
            min_value=0,
            max_value=5_000_000,
            value=1_500_000,
            step=50_000,
        )
        anciennete = st.slider("Ancienneté (mois)", 1, 120, 24)
        type_abonnement = st.selectbox("Type d’abonnement", ["Prépayé", "Postpayé"])

    with c3:
        forfait_international = st.selectbox("Forfait international", ["Oui", "Non"])
        messagerie_vocale = st.selectbox("Messagerie vocale", ["Oui", "Non"])
        minutes_internationales = st.number_input(
            "Minutes internationales / mois",
            min_value=0.0,
            max_value=500.0,
            value=5.0,
            step=1.0,
            format="%.2f",
        )

    st.markdown("### 📱 Usage & comportement")

    u1, u2, u3 = st.columns(3)

    with u1:
        recharge_mensuelle = st.number_input(
            "Recharge mensuelle moyenne (GNF)",
            min_value=0,
            max_value=2_000_000,
            value=200_000,
            step=50_000,
        )
        minutes_jour = st.number_input(
            "Minutes en journée / mois", min_value=0.0, max_value=10_000.0, value=0.0
        )

    with u2:
        minutes_nuit = st.number_input(
            "Minutes de nuit / mois", min_value=0.0, max_value=10_000.0, value=0.0
        )
        donnees_mo = st.number_input(
            "Données Internet (Mo / mois)", min_value=0.0, max_value=100_000.0, value=0.0
        )

    with u3:
        nombre_sms = st.number_input(
            "Nombre de SMS / mois", min_value=0, max_value=10_000, value=0
        )
        appels_service_client = st.number_input(
            "Appels au service client (30 jours)", min_value=0, max_value=100, value=0
        )

    c4, c5, c6 = st.columns(3)
    with c4:
        pannes_signalees_30j = st.number_input(
            "Pannes signalées (30 jours)", min_value=0, max_value=60, value=0
        )
    with c5:
        retard_paiement_jours = st.number_input(
            "Retard de paiement (jours)", min_value=0, max_value=90, value=0
        )

    st.markdown("")

    # ------------------------
    # Bouton de prédiction
    # ------------------------
    predict_btn = st.button("Lancer la prédiction 🔍")

    if predict_btn:
        # Encodage simple des variables catégorielles
        x_vec = np.array(
            [
                REGION_MAP[region],
                SEXE_MAP[sexe],
                age,
                revenu,
                anciennete,
                ABO_MAP[type_abonnement],
                OUI_NON_MAP[forfait_international],
                OUI_NON_MAP[messagerie_vocale],
                recharge_mensuelle,
                minutes_jour,
                minutes_nuit,
                donnees_mo,
                nombre_sms,
                appels_service_client,
                pannes_signalees_30j,
                retard_paiement_jours,
                minutes_internationales,
            ],
            dtype=float,
        ).reshape(1, -1)

        proba_churn = float(model.predict_proba(x_vec)[0, 1])
        pred = int(proba_churn >= 0.5)

        risk_label = "Client à risque de résiliation" if pred == 1 else "Client plutôt fidèle"
        risk_color = "🔴" if pred == 1 else "🟢"

        st.markdown("### 🔎 Résultat de la prédiction")

        m1, m2 = st.columns([1, 2])

        with m1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric(
                label="Probabilité de churn",
                value=f"{proba_churn*100:.1f} %",
                delta="Risque élevé" if pred == 1 else "Risque modéré/FAIBLE",
            )
            st.markdown("</div>", unsafe_allow_html=True)

        with m2:
            st.success(f"{risk_color} {risk_label}")
            st.caption(
                "Interprétation : **1 = client susceptible de résilier**, "
                "**0 = client fidèle**. Le seuil de décision est fixé à 0.5."
            )

    st.markdown("</div>", unsafe_allow_html=True)


# =====================================================
# 4. ONGLET A PROPOS DU MODELE
# =====================================================

with tab_model:
    st.markdown('<div class="main-block">', unsafe_allow_html=True)
    st.subheader("📊 À propos du modèle")

    st.markdown(
        """
- **Algorithme utilisé :** Random Forest Classifier  
- **Objectif :** prédire si un client va **résilier (churn = 1)** ou **rester (churn = 0)**  
- **Variables principales :** usage voix, données, pannes signalées, appels au service client, etc.

Les graphiques ci-dessous illustrent la logique du modèle et l’importance des variables.
        """
    )

    col_a, col_b = st.columns(2)

    with col_a:
        rf_img = IMG_DIR / "random_forest.png"
        if rf_img.exists():
            st.image(str(rf_img), caption="Schéma simplifié d'un Random Forest", use_container_width=True)

        fi_img = IMG_DIR / "data_processing.jpg"
        if fi_img.exists():
            st.image(str(fi_img), caption="Processus global de traitement de données", use_container_width=True)

    with col_b:
        map_img = IMG_DIR / "guinee_map.png"
        if map_img.exists():
            st.image(str(map_img), caption="Contexte : Guinée – clients télécom", use_container_width=True)

        ml_img = IMG_DIR / "ml_models.webp"
        if ml_img.exists():
            st.image(str(ml_img), caption="Famille de modèles de Machine Learning", use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)
