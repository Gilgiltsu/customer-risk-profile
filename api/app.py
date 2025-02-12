from flask import Flask, request, jsonify
import joblib
import pandas as pd
import os
import zipfile

# Initialiser l'application Flask
app = Flask(__name__)

def load_model():
    """Charge automatiquement le fichier .pkl dans le répertoire courant."""
    try:
        # Trouver le fichier .zip dans le répertoire src
        zip_file = [f for f in os.listdir('src') if f.endswith('.zip')]
        if not zip_file:
            raise FileNotFoundError("Aucun fichier .zip trouvé dans le répertoire 'src'.")

        zip_file_path = os.path.join('src', zip_file[0])

        # Extraire le contenu du fichier ZIP
        with zipfile.ZipFile(zip_file_path, 'r') as zipf:
            zipf.extractall("extracted_model")

        # Trouver le fichier .pkl extrait
        pkl_file = [f for f in os.listdir('extracted_model') if f.endswith('.pkl')]
        if not pkl_file:
            raise FileNotFoundError("Aucun fichier .pkl trouvé dans l'archive ZIP.")

        pkl_file_path = os.path.join('extracted_model', pkl_file[0])

        # Charger le modèle
        print(f"✅ Chargement du modèle depuis {pkl_file_path}")
        return joblib.load(pkl_file_path)
    
    except Exception as e:
        print(f"❌ Erreur lors du chargement du modèle : {e}")
        return None

# Charger le modèle
model = load_model()

@app.route('/', methods=['GET'])
def home():
    """Message d'accueil sur la page d'accueil de l'API."""
    return jsonify({
        "message": "Bienvenue sur l'API Customer Risk Profile ! 🚀",
        "status": "API opérationnelle",
        "routes": ["/predict", "/health", "/routes"]
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Point de terminaison pour faire des prédictions."""
    if model is None:
        return jsonify({"error": "Modèle non chargé"}), 500

    try:
        data = request.json  # Attend un JSON avec les caractéristiques
        df = pd.DataFrame(data)
        prediction = model.predict(df)
        return jsonify({'prediction': prediction.tolist()})
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
