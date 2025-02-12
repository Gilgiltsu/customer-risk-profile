import pandas as pd
import numpy as np
import os
import joblib
import mlflow
import mlflow.sklearn
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, recall_score, f1_score
from lightgbm import LGBMClassifier

def load_csv_from_dropbox(shared_link):
    """Charge un fichier CSV directement depuis un lien partagé Dropbox."""
    # Convertir le lien partagé en lien direct
    direct_link = shared_link.replace("dl=0", "dl=1")

    # Charger le fichier CSV dans un DataFrame pandas
    df = pd.read_csv(direct_link)
    return df

def train_model(X, y, best_params):
    """Entraîne le modèle avec les meilleurs hyperparamètres."""
    model = LGBMClassifier(**best_params)
    model.fit(X, y)
    return model

def save_model(model, filename):
    """Sauvegarde le modèle en tant que fichier .pkl."""
    joblib.dump(model, filename)
    print(f"Modèle sauvegardé en tant que {filename}")

def load_model(filename):
    """Charge le modèle depuis un fichier .pkl."""
    model = joblib.load(filename)
    print(f"Modèle chargé depuis {filename}")
    return model

def log_model_to_mlflow(model, run_name, X_test, y_test, signature=None):
    """Enregistre le modèle dans MLflow avec les métriques et caractéristiques."""
    # Spécifier ou créer une expérience
    experiment_name = "Model_Retraining_Experiment"
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=run_name) as run:
        # Enregistrer le modèle
        mlflow.sklearn.log_model(model, "model", registered_model_name="BestModel", signature=signature)

        # Prédire et calculer les métriques
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        # Calculer le score métier
        optimal_threshold, optimal_score = business_score(y_test, y_proba)

        # Enregistrer les métriques
        mlflow.log_metric("auc", roc_auc_score(y_test, y_proba))
        mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))
        mlflow.log_metric("recall", recall_score(y_test, y_pred))
        mlflow.log_metric("f1", f1_score(y_test, y_pred))
        mlflow.log_metric("business_score", optimal_score)
        mlflow.log_param("optimal_threshold", optimal_threshold)

        print(f"Modèle enregistré dans MLflow avec le nom 'BestModel'")

def business_score(y_true, y_proba, cost_fn=10, cost_fp=1):
    """Calcule le score métier."""
    thresholds = np.arange(0, 1, 0.01)
    scores = []

    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        fn = np.sum((y_true == 1) & (y_pred == 0))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        score = (fn * cost_fn) + (fp * cost_fp)
        scores.append((threshold, score))

    # Trouver le seuil optimal
    optimal_threshold, optimal_score = min(scores, key=lambda x: x[1])
    return optimal_threshold, optimal_score

def main(shared_link):
    # Charger les données depuis Dropbox
    df = load_csv_from_dropbox(shared_link)

    # Préparer les données
    X = df.drop('target', axis=1)
    y = df['target']

    # Diviser les données en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Meilleurs hyperparamètres trouvés précédemment
    best_params = {
        'learning_rate': 0.01,
        'n_estimators': 500,
        'num_leaves': 127
    }

    # Entraîner le modèle
    model = train_model(X_train, y_train, best_params)

    # Générer un nom de fichier avec version basée sur la date et l'heure
    version = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_filename = f"model_{version}.pkl"

    # Sauvegarder le modèle
    save_model(model, model_filename)

    # Enregistrer le modèle dans MLflow
    input_example = X_train.sample(1)
    signature = mlflow.models.infer_signature(input_example, model.predict(input_example))
    log_model_to_mlflow(model, run_name=f"Model_Retraining_{version}", X_test=X_test, y_test=y_test, signature=signature)

if __name__ == "__main__":
    # Lien partagé Dropbox vers le fichier de données
    shared_link = "https://www.dropbox.com/scl/fi/ywb34b2q9dafx9ifkt10q/df_cleaned.csv?rlkey=s3f29j4267ef0h2qv77knula1&st=bfeef662&dl=0"

    # Exécuter le processus de réentraînement
    main(shared_link)
