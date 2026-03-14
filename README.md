# 🏥 Analyse des Coûts d'Assurance Médicale — Détection de Biais & Modélisation

Application Streamlit d'analyse d'un dataset d'assurance médicale, avec détection de biais et modélisation prédictive.

## 📋 Problématique

Les charges médicales sont-elles équitables entre les différents groupes démographiques ? Ce projet analyse trois axes de biais potentiels :
- **Genre** : les hommes et les femmes paient-ils des montants équivalents à profil similaire ?
- **Âge** : les seniors sont-ils pénalisés de manière disproportionnée ?
- **Statut fumeur** : l'écart de tarification est-il proportionné au risque réel ?

## 🗂 Structure du projet

```
1_🏠_Accueil.py              # Page d'accueil (KPIs, contexte, aperçu des données)
pages/
  2_📊_Exploration.py        # Exploration interactive avec filtres
  3_⚠_Detection_Biais.py     # Métriques de fairness (parité démographique, impact disproportionné)
  4_🤖_Modelisation.py       # Modélisation (Gradient Boosting, Random Forest) & analyse de fairness
data/
  insurance.csv               # Dataset original
  insurance_clean.csv         # Dataset nettoyé
models/
  gradient_boosting.pkl       # Modèle Gradient Boosting entraîné
  random_forest.pkl           # Modèle Random Forest entraîné
  scaler.pkl                  # Scaler pour normalisation
utils/
  fairness.py                 # Fonctions de métriques de fairness
```

## 🚀 Lancement en local

```bash
pip install -r requirements.txt
streamlit run "1_🏠_Accueil.py"
```

## 🛠 Technologies

- Python, Streamlit
- Pandas, NumPy
- Plotly
- Scikit-learn
