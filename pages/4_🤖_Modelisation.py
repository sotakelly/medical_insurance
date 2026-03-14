import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
from utils.fairness import (
    demographic_parity_difference,
    disparate_impact_ratio,
    group_regression_metrics,
)

st.set_page_config(page_title="Modélisation", page_icon="🤖", layout="wide")


@st.cache_data
def load_data():
    return pd.read_csv("data/insurance.csv")


@st.cache_resource
def load_models():
    gb = joblib.load("models/gradient_boosting.pkl")
    rf = joblib.load("models/random_forest.pkl")
    scaler = joblib.load("models/scaler.pkl")
    return gb, rf, scaler


df_raw = load_data()
gb_model, rf_model, scaler = load_models()

st.title("🤖 Modélisation & Fairness")
st.markdown("---")

# ── Prepare data (same preprocessing as notebook) ──
df = df_raw.copy()
sensitive_original = df["sex"].copy()  # keep original for fairness analysis

df["sex"] = df["sex"].map({"male": 0, "female": 1})
df["smoker"] = df["smoker"].map({"no": 0, "yes": 1})
df = pd.get_dummies(df, columns=["region"], drop_first=True)

X = df.drop("charges", axis=1)
y = df["charges"].values
y_log = np.log1p(y)

# Scale
X_scaled = scaler.transform(X)

# ── Model selection ──
model_name = st.selectbox("Choisir le modèle", ["Gradient Boosting", "Random Forest"])
model = gb_model if model_name == "Gradient Boosting" else rf_model

# Predict (models trained on log scale)
pred_log = model.predict(X_scaled)
pred = np.expm1(pred_log)

# ── 1. Global Performance ──
st.header("📊 Performance Globale")

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

col1, col2, col3 = st.columns(3)
col1.metric("R²", f"{r2_score(y, pred):.4f}")
col2.metric("RMSE", f"${mean_squared_error(y, pred) ** 0.5:,.0f}")
col3.metric("MAE", f"${mean_absolute_error(y, pred):,.0f}")

with st.expander("ℹ️ Comprendre ces métriques"):
    st.markdown("""
    - **R² (Coefficient de détermination)** : mesure la proportion de la variance des charges
      expliquée par le modèle. Varie de 0 à 1 — plus c'est proche de 1, meilleur est le modèle.
    - **RMSE (Root Mean Squared Error)** : erreur moyenne en dollars, pénalisant davantage les
      grosses erreurs. Plus la valeur est basse, mieux c'est.
    - **MAE (Mean Absolute Error)** : erreur moyenne absolue en dollars. Plus intuitive que le RMSE,
      elle représente l'écart moyen entre la prédiction et la réalité.
    """)

st.markdown("---")

# ── 2. Actual vs Predicted ──
st.header("🎯 Prédictions vs Réel")

scatter_df = pd.DataFrame({"Actual": y, "Predicted": pred, "sex": sensitive_original.values})

fig_scatter = px.scatter(
    scatter_df, x="Actual", y="Predicted", color="sex",
    title=f"{model_name} : Charges Réelles vs Prédites",
    labels={"Actual": "Charges Réelles ($)", "Predicted": "Charges Prédites ($)"},
    opacity=0.5,
)
# Perfect prediction line
max_val = max(y.max(), pred.max())
fig_scatter.add_trace(
    go.Scatter(x=[0, max_val], y=[0, max_val], mode="lines",
               line=dict(color="red", dash="dash"), name="Prédiction parfaite")
)
st.plotly_chart(fig_scatter, use_container_width=True)

with st.expander("ℹ️ Comment lire ce graphique ?"):
    st.markdown("""
    Chaque point représente un assuré : son **coût réel** (axe X) vs le **coût prédit** par le modèle (axe Y).

    - **Ligne rouge pointillée** = prédiction parfaite. Si le modèle était parfait, tous les points seraient alignés sur cette ligne.
    - **Points proches de la ligne** → le modèle prédit bien ces cas.
    - **Points éloignés de la ligne** → erreurs de prédiction (le modèle sous-estime ou sur-estime).
    - **Points au-dessus de la ligne** → le modèle **sur-estime** les charges.
    - **Points en dessous de la ligne** → le modèle **sous-estime** les charges.
    - Les couleurs distinguent les **hommes** et **femmes** pour repérer un éventuel biais visuel entre genres.
    """)

st.markdown("---")

# ── 3. Fairness metrics on predictions ──
st.header("⚖️ Métriques de Fairness sur les Prédictions")

sensitive_attr = st.selectbox(
    "Attribut sensible", ["sex", "age_group", "smoker"],
    format_func=lambda x: {"sex": "👤 Genre", "age_group": "📅 Âge (tranches)", "smoker": "🚬 Statut fumeur"}[x],
    index=0, key="fair_attr",
)

# Create age groups
df_raw["age_group"] = pd.cut(df_raw["age"], bins=[17, 30, 45, 60, 100],
                              labels=["18-30", "31-45", "46-60", "60+"])

attr_values = df_raw[sensitive_attr].values

dp_result = demographic_parity_difference(y_true=y, y_pred=pred, sensitive_attribute=attr_values)

unpriv_map = {"sex": "female", "age_group": "60+", "smoker": "yes"}
priv_map = {"sex": "male", "age_group": "18-30", "smoker": "no"}

di_result = disparate_impact_ratio(
    y_true=y, y_pred=pred, sensitive_attribute=attr_values,
    unprivileged_value=unpriv_map[sensitive_attr],
    privileged_value=priv_map[sensitive_attr],
)

col1, col2 = st.columns(2)
col1.metric(
    "Écart de Parité (prédictions moyennes)",
    f"${dp_result['difference']:,.0f}",
    help="Différence entre les charges moyennes prédites du groupe le plus élevé et le plus bas.",
)
col2.metric(
    "Ratio d'Impact Disproportionné",
    f"{di_result['ratio']:.3f}",
    help="Ratio des charges moyennes prédites entre groupes. 1.0 = équité parfaite.",
)

st.markdown("---")

# ── 4. Performance par groupe sensible ──
st.header("📋 Performance par Groupe")

group_metrics = group_regression_metrics(y, pred, attr_values)

metrics_df = pd.DataFrame(group_metrics).T
metrics_df.index.name = sensitive_attr

st.dataframe(
    metrics_df.style.format({
        "n": "{:,}",
        "R²": "{:.4f}",
        "RMSE": "${:,.0f}",
        "MAE": "${:,.0f}",
        "Mean Actual": "${:,.0f}",
        "Mean Predicted": "${:,.0f}",
    }),
    use_container_width=True,
)

# Bar chart of metrics by group
attr_label_map = {"sex": "Genre", "age_group": "Tranche d'âge", "smoker": "Statut fumeur"}
fig_group = px.bar(
    metrics_df.reset_index().melt(
        id_vars=[sensitive_attr],
        value_vars=["Mean Actual", "Mean Predicted"],
    ),
    x=sensitive_attr, y="value", color="variable", barmode="group",
    title=f"Charges Moyennes Réelles vs Prédites par {attr_label_map[sensitive_attr]}",
    labels={"value": "Charges ($)", "variable": "Type"},
    text_auto="$,.0f",
)
fig_group.update_traces(textposition="outside")
st.plotly_chart(fig_group, use_container_width=True)

st.markdown("---")

# ── 5. Residual analysis by group ──
st.header("📉 Résidus par Groupe")

residuals_df = pd.DataFrame({
    sensitive_attr: attr_values,
    "Résidu": y - pred,
})

fig_resid = px.box(
    residuals_df, x=sensitive_attr, y="Résidu", color=sensitive_attr,
    title=f"Distribution des Résidus par {attr_label_map[sensitive_attr]}",labels={"Résidu": "Résidu ($)"},
)
fig_resid.add_hline(y=0, line_dash="dash", line_color="red")
st.plotly_chart(fig_resid, use_container_width=True)

# ── Interpretation ──
st.header("💡 Interprétation")

ratio_model = di_result["ratio"]
normalized_ratio_model = min(ratio_model, 1 / ratio_model) if ratio_model != 0 else 0

if sensitive_attr == "sex":
    if normalized_ratio_model > 0.9:
        ratio_model_label = "proche de 1 → écart faible entre genres"
    elif normalized_ratio_model > 0.8:
        ratio_model_label = "entre 0.8 et 0.9 → écart notable entre genres"
    else:
        ratio_model_label = f"ratio normalisé = {normalized_ratio_model:.3f} → **biais significatif** selon la règle des 80 %"
elif sensitive_attr == "smoker":
    ratio_model_label = f"ratio de {ratio_model:.1f}× — écart élevé mais **attendu** car le tabagisme est un facteur de risque médical avéré"
else:  # age_group
    ratio_model_label = f"ratio de {ratio_model:.1f}× — écart **attendu** car l'âge est un facteur actuariel reconnu"

st.markdown(f"""
**Analyse du modèle {model_name} :**

- Le modèle atteint un **R² de {r2_score(y, pred):.4f}**, expliquant environ
  {r2_score(y, pred) * 100:.1f}% de la variance des charges.
- Le ratio d'impact disproportionné sur les **prédictions** est de **{ratio_model:.3f}**
  ({ratio_model_label}).
- Les résidus par groupe permettent de vérifier si le modèle sur-estime ou sous-estime
  systématiquement les charges d'un groupe par rapport à l'autre.
""")

# Recommandation contextuelle basée sur les résultats réels
# Comparer écart réel vs écart prédit
actual_means = {}
pred_means = {}
for group in set(attr_values):
    mask = attr_values == group
    actual_means[group] = y[mask].mean()
    pred_means[group] = pred[mask].mean()

actual_gap = max(actual_means.values()) - min(actual_means.values())
pred_gap = max(pred_means.values()) - min(pred_means.values())
amplification = pred_gap - actual_gap

if sensitive_attr == "sex":
    if abs(amplification) < 200:
        st.success("✅ Pas d'amplification du biais par le modèle")
        st.markdown(
            "L'écart de charges entre genres dans les données réelles est de "
            f"{actual_gap:,.0f} \\$, et dans les prédictions de {pred_gap:,.0f} \\$ "
            f"(différence de {amplification:+,.0f} \\$). "
            "Le modèle reproduit fidèlement l'écart observé sans l'amplifier.")
    elif amplification > 0:
        st.error("🚨 Le modèle amplifie le biais de genre")
        st.markdown(
            "L'écart de charges entre genres dans les données réelles est de "
            f"{actual_gap:,.0f} \\$, mais le modèle produit un écart de {pred_gap:,.0f} \\$ "
            f"(+{amplification:,.0f} \\$). "
            "Le modèle sur-pénalise un genre par rapport à l'autre au-delà de ce que les données justifient.")
    else:
        st.success("✅ Le modèle réduit l'écart de genre")
        st.markdown(
            "L'écart de charges entre genres dans les données réelles est de "
            f"{actual_gap:,.0f} \\$, et le modèle le réduit à {pred_gap:,.0f} \\$ "
            f"({amplification:+,.0f} \\$).")
elif sensitive_attr == "smoker":
    if abs(amplification) / actual_gap < 0.1:
        st.success("✅ Écart proportionné au risque réel")
        st.markdown(
            "L'écart fumeurs/non-fumeurs dans les données réelles est de "
            f"{actual_gap:,.0f} \\$, et dans les prédictions de {pred_gap:,.0f} \\$ "
            f"(différence de {amplification:+,.0f} \\$). "
            "Le modèle ne sur-pénalise pas les fumeurs — l'écart est fidèle au surcoût médical réel.")
    else:
        st.warning("⚠️ Écart légèrement disproportionné")
        st.markdown(
            "L'écart fumeurs/non-fumeurs dans les données réelles est de "
            f"{actual_gap:,.0f} \\$, mais le modèle produit un écart de {pred_gap:,.0f} \\$ "
            f"(différence de {amplification:+,.0f} \\$). "
            "Le modèle amplifie légèrement la pénalisation des fumeurs par rapport au surcoût réel.")
elif sensitive_attr == "age_group":
    if abs(amplification) / actual_gap < 0.1:
        st.success("✅ Écart proportionné au facteur âge")
        st.markdown(
            "L'écart entre tranches d'âge dans les données réelles est de "
            f"{actual_gap:,.0f} \\$, et dans les prédictions de {pred_gap:,.0f} \\$ "
            f"(différence de {amplification:+,.0f} \\$). "
            "Le modèle ne produit pas de discrimination disproportionnée liée à l'âge.")
    else:
        st.warning("⚠️ Écart légèrement disproportionné")
        st.markdown(
            "L'écart entre tranches d'âge dans les données réelles est de "
            f"{actual_gap:,.0f} \\$, mais le modèle produit un écart de {pred_gap:,.0f} \\$ "
            f"(différence de {amplification:+,.0f} \\$). "
            "Le modèle amplifie légèrement les différences liées à l'âge.")
