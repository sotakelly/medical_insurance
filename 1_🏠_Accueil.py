import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="Medical Insurance - Détection de Biais",
    page_icon="🏥",
    layout="wide",
)

# ── Load data ──
@st.cache_data
def load_data():
    return pd.read_csv("data/insurance.csv")

df = load_data()

# ══════════════════════════════════════════════
# PAGE 1 : 🏠 Accueil
# ══════════════════════════════════════════════

st.title("🏥 Analyse des Coûts d'Assurance Médicale")
st.markdown("### Détection de Biais & Modélisation")
st.markdown("---")

# ── Contexte et Problématique ──
st.header("📋 Contexte et Problématique")

st.markdown("""
Ce projet analyse un dataset d'assurance médicale contenant des informations sur les assurés
(âge, sexe, IMC, nombre d'enfants, statut tabagique et région) ainsi que les **charges médicales**
facturées par l'assureur. L'objectif principal est de **prédire les coûts d'assurance médicale**
et de détecter d'éventuels **biais** dans la tarification.

La question centrale est la suivante : **les charges médicales sont-elles équitables entre les
différents groupes démographiques ?** Nous analysons trois axes de biais potentiels :
- **Le genre** : les hommes et les femmes paient-ils des montants équivalents à profil similaire ?
- **L'âge** : les seniors sont-ils pénalisés de manière disproportionnée ?
- **Le statut fumeur** : l'écart de tarification est-il proportionné au risque réel ?

Un écart significatif pourrait indiquer un biais systémique dans la détermination des primes,
avec des conséquences réelles pour les assurés.

Pour répondre à ces questions, nous explorons les données, entraînons des modèles prédictifs
(Gradient Boosting, Random Forest) et appliquons des **métriques de fairness**
(parité démographique, impact disproportionné) afin de quantifier et visualiser les biais.
""")

st.markdown("---")

# ── 4 KPIs ──
st.header("📊 Métriques Clés")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Nombre de lignes", f"{df.shape[0]:,}")
col2.metric("Nombre de colonnes", f"{df.shape[1]}")

missing_rate = df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100
col3.metric("Valeurs manquantes", f"{missing_rate:.1f} %")

col4.metric("Charges moyennes", f"${df['charges'].mean():,.0f}")

st.markdown("---")

# ── Distribution de la variable cible ──
st.header("📈 Distribution de la Variable Cible")
col_left, col_right = st.columns(2)
with col_left:
    st.markdown("**Statistiques des charges**")
    st.dataframe(
        df["charges"].describe().to_frame().style.format("${:,.2f}"),
    )
with col_right:
    import plotly.express as px
    fig = px.histogram(
        df, x="charges", nbins=40, title="Distribution des Charges Médicales",
        labels={"charges": "Charges ($)", "count": "Fréquence"},
        color_discrete_sequence=["#636EFA"],
    )
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# ── Aperçu des données ──
st.header("🔍 Aperçu des Données")
st.dataframe(df, use_container_width=True, height=400)

st.markdown("---")

# ── Description des colonnes ──
st.header("📝 Description des Colonnes")

column_descriptions = {
    "age": "Âge de l'assuré (entier)",
    "sex": "Sexe de l'assuré (male / female)",
    "bmi": "Indice de Masse Corporelle (IMC) — mesure du poids par rapport à la taille",
    "children": "Nombre d'enfants/dépendants couverts par l'assurance",
    "smoker": "Statut tabagique de l'assuré (yes / no)",
    "region": "Région de résidence aux États-Unis (southeast, southwest, northeast, northwest)",
    "charges": "Montant des frais médicaux facturés par l'assurance ($) — **variable cible**",
}

desc_df = pd.DataFrame(
    [{"Colonne": k, "Type": str(df[k].dtype), "Description": v}
     for k, v in column_descriptions.items()]
)
st.table(desc_df)
