# 🏦 Système de Credit Scoring End-to-End & MLOps

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.109-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Enabled-2496ED?style=for-the-badge&logo=docker&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)

Ce projet implémente une solution complète de **Credit Scoring** pour prédire le risque de défaut de paiement de clients (basé sur le dataset *Home Credit Default Risk*).

L'objectif est de démontrer une approche **MLOps** rigoureuse, allant de l'ingénierie des données brutes à la mise en production d'une API REST conteneurisée.



## 🚀 Fonctionnalités Clés

* **⚡ Pipeline ETL Optimisé :** Conversion des données brutes en format **Parquet**, réduisant le temps de chargement et le stockage (gain de performance x2.3 constaté).
* **🤖 Modélisation Avancée :** Entraînement d'un **Random Forest** avec gestion du déséquilibre des classes (`class_weight='balanced'`) pour maximiser le ROC-AUC.
* **🛡️ Pipeline Robuste :** Utilisation de `scikit-learn Pipeline` et `ColumnTransformer` pour encapsuler le pré-traitement (imputation, scaling, encoding) et éviter le *training-serving skew*.
* **🔌 API REST (FastAPI) :** API asynchrone exposant le modèle pour des prédictions en temps réel, avec validation des données via Pydantic.
* **🐳 Conteneurisation (Docker) :** Environnement isolé et reproductible, prêt pour un déploiement Cloud (AWS/Azure/GCP).



## 🛠️ Stack Technique

| Catégorie       | Technologies                                 |
|:----------------|:---------------------------------------------|
| **Langage**     | Python 3.10+                                 |
| **Data & ML**   | Pandas, Scikit-Learn, Joblib, PyArrow, NumPy |
| **API Backend** | FastAPI, Uvicorn, Pydantic                   |
| **DevOps**      | Docker, Git                                  |



## ⚡ Installation & Démarrage

### Option 1 : Via Docker (Recommandé)
C'est la méthode la plus fiable pour exécuter le projet dans un environnement stable, identique à la production.

1. **Construire l'image Docker :**
```bash
docker build -t credit-scoring-api .
```

2. **Lancer le conteneur**
```bash
docker run -p 8000:8000 credit-scoring-api
```

### Option 2 : En local (Sans Docker)
Pré-requis : Python 3.10 ou supérieur.

1. **Cloner le projet et installer les dépendances :**
```bash
git clone [https://github.com/321Coder123/credit-scoring-mlops.git](https://github.com/VOTRE_PSEUDO/credit-scoring-mlops.git)
cd credit-scoring

# Création de l'environnement virtuel
python -m venv .venv

# Activation (Windows)
.venv\Scripts\activate
# Activation (Mac/Linux)
source .venv/bin/activate

# Installation des librairies
pip install -r requirements.txt
```

2. **Lancer le pipeline d'entraînement (Optionnel) : Si le modèle n'est pas présent dans le dossier models/, relancez l'entraînement :**
```bash
python -m src.model
```

3. **Démarrer le serveur API :**
```bash
uvicorn api.app:app --reload
```



## 📂 Structure du Projet

```text
├── api/             # Code de l'API (FastAPI) & Schémas Pydantic
│   ├── app.py       # Point d'entrée de l'application
│   └── schemas.py   # Définition des modèles de données
├── data/            # Données (ignorées par Git)
│   ├── raw/         # Données brutes (CSV)
│   └── processed/   # Données transformées (Parquet)
├── models/          # Modèle entraîné (.joblib)
├── notebooks/       # Explorations (EDA) et Benchmarks
├── src/             # Code source
│   ├── data_loader.py    # Scripts ETL
│   ├── preprocessing.py  # Pipelines de nettoyage
│   └── model.py          # Entraînement et évaluation
├── Dockerfile       # Configuration de l'image Docker
├── requirements.txt # Dépendances de production
└── README.md        # Documentation du projet
```