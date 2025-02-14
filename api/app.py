from flask import Flask, request, jsonify
import joblib
import pandas as pd
import os

# Initialiser l'application Flask
app = Flask(__name__)

def load_model():
    """Charge automatiquement le fichier .pkl dans le répertoire 'src'."""
    try:
        # Trouver le fichier .pkl dans le répertoire src
        pkl_file = [f for f in os.listdir('src') if f.endswith('.pkl')]
        if not pkl_file:
            raise FileNotFoundError("Aucun fichier .pkl trouvé dans le répertoire 'src'.")

        pkl_file_path = os.path.join('src', pkl_file[0])

        # Charger le modèle
        model = joblib.load(pkl_file_path)
        print(f"✅ Chargement du modèle depuis {pkl_file_path}")

        return model

    except Exception as e:
        print(f"❌ Erreur lors du chargement du modèle : {e}")
        return None

# Charger le modèle
model = load_model()
optimal_threshold = 0.5 #est renvoyé lors de l'entrainement du model

@app.route('/', methods=['GET'])
def home():
    """Message d'accueil sur la page d'accueil de l'API."""
    return jsonify({
        "message": "Bienvenue sur l'API Customer Risk Profile ! 🚀",
        "status": "API opérationnelle",
        "routes": ["/predict", "/health", "/routes"]
    })

@app.route('/predict', methods=['POST'])
@app.route('/predict', methods=['POST'])
def predict():
    """Point de terminaison pour faire des prédictions."""
    if model is None:
        return jsonify({"error": "Modèle non chargé"}), 500

    try:
        data = request.json  # Attend un JSON avec les caractéristiques
        df = pd.DataFrame(data)

        # Faire une prédiction avec le modèle
        y_proba = model.predict_proba(df)[:, 1]
        y_pred_optimal = [1 if prob >= optimal_threshold else 0 for prob in y_proba]

        print("Probabilités et prédictions optimales :")
        for i, sk_id_curr in enumerate(df['sk_id_curr']):
            print(f"Client {sk_id_curr}: Probabilité = {y_proba[i]:.4f}, Prédiction = {y_pred_optimal[i]}")

        return jsonify({'prediction': y_pred_optimal})
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/health', methods=['GET'])
def health_check():
    """Point de terminaison pour vérifier l'état de l'API."""
    return jsonify({'status': 'healthy'})

@app.route('/routes', methods=['GET'])
def list_routes():
    """Affiche toutes les routes disponibles dans l'API."""
    routes = []
    for rule in app.url_map.iter_rules():
        routes.append(str(rule))
    return jsonify({"available_routes": routes})

if __name__ == '__main__':
    app.run(debug=True)
