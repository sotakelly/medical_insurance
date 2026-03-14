import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(page_title="Exploration des Données", page_icon="📊", layout="wide")


@st.cache_data
def load_data():
    return pd.read_csv("data/insurance.csv")


df = load_data()

st.title("📊 Exploration des Données")
st.markdown("---")

# ── Filtres interactifs (BONUS) ──
st.sidebar.header("🔧 Filtres")

sex_filter = st.sidebar.multiselect("Sexe", df["sex"].unique(), default=df["sex"].unique())
smoker_filter = st.sidebar.multiselect("Fumeur", df["smoker"].unique(), default=df["smoker"].unique())
region_filter = st.sidebar.multiselect("Région", df["region"].unique(), default=df["region"].unique())
age_range = st.sidebar.slider("Âge", int(df["age"].min()), int(df["age"].max()),
                               (int(df["age"].min()), int(df["age"].max())))

df_filtered = df[
    (df["sex"].isin(sex_filter))
    & (df["smoker"].isin(smoker_filter))
    & (df["region"].isin(region_filter))
    & (df["age"].between(age_range[0], age_range[1]))
]

# ── 4 KPIs ──
st.header("📈 Indicateurs Clés")
col1, col2, col3, col4 = st.columns(4)

col1.metric("Total lignes", f"{len(df_filtered):,}")
col2.metric("Charges moyennes", f"${df_filtered['charges'].mean():,.0f}")
col3.metric("Charges médianes", f"${df_filtered['charges'].median():,.0f}")
col4.metric("% Fumeurs", f"{df_filtered['smoker'].value_counts(normalize=True).get('yes', 0) * 100:.1f} %")

st.markdown("---")

# ── Visualisation 1 : Distribution de la variable cible ──
st.header("1️⃣ Distribution de la Variable Cible")

fig1 = px.histogram(
    df_filtered, x="charges", nbins=40,
    title="Distribution des Charges Médicales",
    labels={"charges": "Charges ($)", "count": "Fréquence"},
    color_discrete_sequence=["#636EFA"],
    marginal="box",
)
st.plotly_chart(fig1, use_container_width=True)

st.markdown("---")

# ── Visualisation 2 : Comparaison entre groupes (attribut sensible) ──
st.header("2️⃣ Comparaison des Charges par Groupe")

group_col = st.selectbox("Attribut de comparaison", ["sex", "smoker", "region"])

group_stats = df_filtered.groupby(group_col)["charges"].mean().reset_index()
group_stats.columns = [group_col, "Charges Moyennes"]

fig2 = px.bar(
    group_stats, x=group_col, y="Charges Moyennes", color=group_col,
    title=f"Charges Moyennes par {group_col.capitalize()}",
    labels={"Charges Moyennes": "Charges Moyennes ($)"},
    text_auto="$.0f",
)
fig2.update_traces(textposition="outside")
st.plotly_chart(fig2, use_container_width=True)

st.markdown("---")

# ── Visualisation 3 : Scatter plot + Box plot + Heatmap ──
st.header("3️⃣ Visualisations Supplémentaires")

tab1, tab2, tab3 = st.tabs(["Scatter Plot", "Box Plot", "Heatmap Corrélations"])

with tab1:
    fig3 = px.scatter(
        df_filtered, x="age", y="charges", color="smoker",
        title="Charges vs Âge (coloré par statut fumeur)",
        labels={"age": "Âge", "charges": "Charges ($)"},
        opacity=0.6,
    )
    st.plotly_chart(fig3, use_container_width=True)

with tab2:
    box_group = st.selectbox("Grouper par", ["sex", "smoker", "region"], key="box_group")
    fig4 = px.box(
        df_filtered, x=box_group, y="charges", color=box_group,
        title=f"Distribution des Charges par {box_group.capitalize()}",
        labels={"charges": "Charges ($)"},
    )
    st.plotly_chart(fig4, use_container_width=True)

with tab3:
    numeric_df = df_filtered.select_dtypes(include=np.number)
    corr = numeric_df.corr()
    fig5 = px.imshow(
        corr, text_auto=".2f", color_continuous_scale="RdBu_r",
        title="Matrice de Corrélation",
        zmin=-1, zmax=1,
    )
    st.plotly_chart(fig5, use_container_width=True)
