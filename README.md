# Projet de Modèle de Scoring de Prédiction de recouvrement
## Mission

Votre mission est de construire un modèle de scoring qui prédit la probabilité de recouvrement d'un client de manière automatique. Ce projet vise à :

1. **Construire un modèle de scoring** : Développer un modèle capable de prédire la probabilité de remboursement d'un client.
2. **Analyser les features** : Identifier les caractéristiques (features) qui contribuent le plus au modèle, tant au niveau global qu'au niveau local pour chaque client, afin de fournir une transparence sur le score attribué.
3. **Mettre en production le modèle** : Déployer le modèle via une API et créer une interface de test pour cette API.

## Arborescence du Projet

Voici une vue d'ensemble de l'architecture du projet :

```
customer-risk-profile/
│
├── .github/
│   └── workflows/
│       └── deploy.yml               # Configuration GitHub Actions pour le déploiement continu
│
├── api/
│   ├── app.py                       # Code de l'API Flask
│   ├── Procfile                     # Configuration pour Heroku
│   ├── requirements.txt              # Dépendances Python
│   └── test_api.py                  # Tests unitaires pour l'API
│
├── src/
│   ├── model_20250211_191632.zip    # Modèle sérialisé, zipper car heroku ne gere pas bien les lfs
│   └── train.py                     # Script pour entraîner et sauvegarder le modèle
│
├── README.md                        # Documentation du projet
├── runtime.txt                     # Spécifie la version de Python
└── setup_mlflow.sh                  # Script de configuration pour MLflow
```

## Description des Composants

- **`.github/workflows/deploy.yml`** : Fichier de configuration pour GitHub Actions qui automatise le déploiement continu de l'API sur Heroku.

- **`api/`** : Contient les fichiers nécessaires pour l'API Flask :
  - **`app.py`** : Code principal de l'API Flask qui charge le modèle et gère les requêtes de prédiction.
  - **`Procfile`** : Configuration pour Heroku indiquant comment démarrer l'application.
  - **`requirements.txt`** : Liste des dépendances Python nécessaires pour exécuter l'application.
  - **`test_api.py`** : Tests unitaires pour vérifier le bon fonctionnement de l'API.

- **`mlruns/`** : Répertoire où MLflow stocke les résultats des expériences et des modèles.

- **`src/`** : Contient les fichiers de données et les scripts pour l'entraînement du modèle :
  - **`model_20250211_191632.zip`** : Modèle entraîné et sérialisé.
  - **`train.py`** : Script pour entraîner le modèle sur les données et le sauvegarder.

- **`runtime.txt`** : Fichier spécifiant la version de Python à utiliser.

- **`setup_mlflow.sh`** : Script de configuration pour MLflow, utilisé pour configurer l'environnement MLflow.

## Instructions

1. **Cloner le dépôt** : Utilisez `git clone` pour cloner ce dépôt sur votre machine locale.
2. **Configurer l'environnement** : Installez les dépendances avec `pip install -r api/requirements.txt`.
3. **Exécuter les tests** : Lancez les tests unitaires avec `pytest`.
4. **Déployer sur Heroku** : Poussez les modifications vers GitHub pour déclencher le déploiement automatique sur Heroku.

## Objectifs

- **Automatisation** : Automatiser la prédiction de la probabilité de recouvrement des clients.
- **Transparence** : Fournir une analyse des features pour une meilleure compréhension des scores attribués.
- **Déploiement** : Mettre en production le modèle via une API pour une utilisation pratique.

Ce projet vise à fournir une solution complète pour la prédiction de la recouvrement des clients, avec un accent sur la transparence et l'automatisation.

