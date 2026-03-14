import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from utils.fairness import demographic_parity_difference, disparate_impact_ratio

st.set_page_config(page_title="Détection de Biais", page_icon="⚠", layout="wide")


@st.cache_data
def load_data():
    return pd.read_csv("data/insurance.csv")


df = load_data()

st.title("⚠ Détection de Biais")
st.markdown("---")

# ── 1. Explication du biais analysé ──
st.header("🔍 Attribut Sensible Analysé")

st.info("**Problématique** : Prédire les coûts d'assurance médicale — "
        "**Biais à analyser** : Genre, Âge, Statut fumeur")

sensitive_attr = st.selectbox(
    "Choisir l'attribut sensible à analyser",
    ["sex", "age_group", "smoker"],
    format_func=lambda x: {"sex": "👤 Genre (sex)", "age_group": "📅 Âge (tranches)", "smoker": "🚬 Statut fumeur"}[x],
    index=0,
)

# Create age groups for bias analysis
df["age_group"] = pd.cut(df["age"], bins=[17, 30, 45, 60, 100],
                          labels=["18-30", "31-45", "46-60", "60+"])

if sensitive_attr == "sex":
    st.markdown("""
    **Attribut sensible : le genre**

    Le genre est un attribut protégé dans de nombreuses législations sur l'assurance.
    Facturer des primes différentes selon le sexe peut constituer une **discrimination directe**,
    même si des différences statistiques existent dans les données historiques.
    Il est essentiel de vérifier si les charges médicales présentent un écart significatif
    entre hommes et femmes, et si un modèle prédictif reproduit ou amplifie cet écart.
    """)
    unprivileged, privileged = "female", "male"
elif sensitive_attr == "age_group":
    st.markdown("""
    **Attribut sensible : l'âge (par tranches)**

    L'âge est un facteur de risque naturel en assurance, mais des écarts de tarification
    disproportionnés peuvent pénaliser injustement les **seniors**. Si les charges augmentent
    de manière non linéaire avec l'âge, cela peut refléter un biais systémique dans la
    tarification plutôt qu'une réalité médicale proportionnée.

    Un modèle qui amplifie cet écart au-delà du risque réel constitue une forme de
    **discrimination par l'âge** (âgisme).
    """)
    unprivileged, privileged = "60+", "18-30"
else:
    st.markdown("""
    **Attribut sensible : le statut fumeur**

    Le tabagisme est un facteur de risque médical avéré, mais des **écarts de tarification
    excessifs** peuvent soulever des questions d'équité. Si le modèle amplifie les écarts
    au-delà de ce que les coûts médicaux réels justifient, cela peut constituer une forme
    de **discrimination disproportionnée**.

    Il est important de vérifier si la pénalisation des fumeurs est proportionnée au
    surcoût médical réel.
    """)
    unprivileged, privileged = "yes", "no"

st.markdown("---")

# ── 2. Métriques de fairness ──
st.header("📏 Métriques de Fairness")

charges = df["charges"].values
attr = df[sensitive_attr].values

# Metric 1: Demographic Parity Difference
dp_result = demographic_parity_difference(
    y_true=charges, y_pred=charges, sensitive_attribute=attr
)

# Metric 2: Disparate Impact Ratio
di_result = disparate_impact_ratio(
    y_true=charges, y_pred=charges, sensitive_attribute=attr,
    unprivileged_value=unprivileged, privileged_value=privileged,
)

col1, col2 = st.columns(2)

col1.metric(
    "Écart de Parité (charges moyennes)",
    f"${dp_result['difference']:,.0f}",
    help="Différence entre les charges moyennes du groupe le plus élevé et le plus bas. "
         "0 = parité parfaite.",
)

col2.metric(
    "Ratio d'Impact Disproportionné",
    f"{di_result['ratio']:.3f}",
    help="Ratio des charges moyennes entre groupes. "
         "1.0 = parfaite équité. En dessous de 0.8 = biais significatif.",
)

st.markdown("---")

# ── 3. Visualisation des résultats ──
st.header("📊 Visualisation des Écarts")

group_means = df.groupby(sensitive_attr)["charges"].agg(["mean", "median", "count"]).reset_index()
group_means.columns = [sensitive_attr, "Charges Moyennes", "Charges Médianes", "Effectif"]

col_left, col_right = st.columns(2)

with col_left:
    fig1 = px.bar(
        group_means, x=sensitive_attr, y="Charges Moyennes", color=sensitive_attr,
        title=f"Charges Moyennes par {sensitive_attr.capitalize()}",
        text_auto="$,.0f",
    )
    fig1.update_traces(textposition="outside")
    st.plotly_chart(fig1, use_container_width=True)

with col_right:
    fig2 = px.box(
        df, x=sensitive_attr, y="charges", color=sensitive_attr,
        title=f"Distribution des Charges par {sensitive_attr.capitalize()}",
        labels={"charges": "Charges ($)"},
    )
    st.plotly_chart(fig2, use_container_width=True)

# Detail table
st.subheader("Détail par groupe")
st.dataframe(
    group_means.style.format({
        "Charges Moyennes": "${:,.2f}",
        "Charges Médianes": "${:,.2f}",
        "Effectif": "{:,}",
    }),
    use_container_width=True,
)

st.markdown("---")

# ── 4. Interprétation ──
st.header("💡 Interprétation")

ratio = di_result["ratio"]
diff = dp_result["difference"]
means = dp_result["group_means"]

higher_group = max(means, key=means.get)
lower_group = min(means, key=means.get)

# Normalize ratio so it's always <= 1 for interpretation (min/max)
normalized_ratio = min(ratio, 1 / ratio) if ratio != 0 else 0

attr_label = {"sex": "le genre", "age_group": "l'âge", "smoker": "le statut fumeur"}[sensitive_attr]

if sensitive_attr == "sex":
    if normalized_ratio > 0.9:
        ratio_interpretation = "proche de 1, indiquant un écart relativement faible entre les genres"
    elif normalized_ratio > 0.8:
        ratio_interpretation = "entre 0.8 et 0.9, indiquant un écart notable entre les genres"
    else:
        ratio_interpretation = f"bien en dessous de 0.8 (ratio normalisé = {normalized_ratio:.3f}), indiquant un **biais significatif** selon la règle des 80 %"
elif sensitive_attr == "smoker":
    ratio_interpretation = f"le groupe **{higher_group}** paie {ratio:.1f}× plus — cet écart est **attendu** car le tabagisme est un facteur de risque médical avéré"
else:  # age_group
    ratio_interpretation = f"le groupe **{higher_group}** paie {ratio:.1f}× plus — cet écart est **attendu** car l'âge est un facteur actuariel reconnu"

st.markdown(f"""
**Résultats de l'analyse de biais sur {attr_label} :**

- Les charges moyennes du groupe **{higher_group}** sont supérieures de **${diff:,.0f}** à celles
  du groupe **{lower_group}**.
- Le ratio d'impact disproportionné est de **{ratio:.3f}** (le groupe **{higher_group}** paie
  {ratio:.1f}× plus que le groupe **{lower_group}**) — {ratio_interpretation}.
- Le groupe **{lower_group}** est relativement **favorisé** avec des charges plus basses,
  tandis que le groupe **{higher_group}** est **pénalisé** avec des charges plus élevées.
""")

# Recommandation contextuelle selon l'attribut
if sensitive_attr == "sex":
    st.warning("""
    **⚠️ Attribut protégé — Vigilance requise**

    Le genre est un **attribut protégé** dans de nombreuses législations sur l'assurance.
    Un écart de tarification entre hommes et femmes, même faible, peut constituer une
    **discrimination directe**. Il est essentiel de vérifier si cet écart persiste après
    contrôle des variables confondantes (âge, IMC, tabagisme). Si le modèle prédictif
    reproduit ou amplifie cet écart, des techniques de **débiaisage** (re-pondération,
    contraintes de fairness) doivent être envisagées.
    """)
elif sensitive_attr == "smoker":
    st.success("""
    **✅ Facteur de risque légitime**

    Le tabagisme est un **facteur de risque médical avéré** : les fumeurs génèrent
    des coûts de santé significativement plus élevés. L'écart de tarification observé
    reflète donc une **réalité médicale**, pas un biais injuste. Cependant, il convient
    de vérifier que le modèle prédictif ne **sur-pénalise** pas les fumeurs au-delà du
    surcoût médical réel (en comparant l'écart dans les données vs dans les prédictions).
    """)
elif sensitive_attr == "age_group":
    st.info("""
    **ℹ️ Facteur actuariel reconnu**

    L'âge est un **facteur de risque naturel** en assurance : les coûts médicaux augmentent
    avec l'âge. L'écart observé est donc médicalement attendu. Toutefois, il faut s'assurer
    que le modèle ne produit pas un écart **disproportionné** par rapport aux coûts réels.
    Si les prédictions amplifient l'écart au-delà des données observées, cela pourrait
    constituer une forme de **discrimination par l'âge** (âgisme).
    """)
