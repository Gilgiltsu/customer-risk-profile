from flask import Flask, request, jsonify
import joblib
import pandas as pd
import os
import zipfile

# Initialiser l'application Flask
app = Flask(__name__)

def load_model():
    """Charge automatiquement le fichier .pkl dans le répertoire courant."""
    # Trouver le fichier .zip dans le répertoire src
    zip_file = [f for f in os.listdir('src') if f.endswith('.zip')]
    if not zip_file:
        raise FileNotFoundError("Aucun fichier .zip trouvé dans le répertoire courant.")

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
    print(f"Chargement du modèle depuis {pkl_file_path}")
    return joblib.load(pkl_file_path)

# Charger le modèle
model = load_model()

@app.route('/predict', methods=['POST'])
def predict():
    """Point de terminaison pour faire des prédictions."""
    data = request.json  # Attend un JSON avec les caractéristiques
    df = pd.DataFrame(data)
    prediction = model.predict(df)
    return jsonify({'prediction': prediction.tolist()})

@app.route('/health', methods=['GET'])
def health_check():
    """Point de terminaison pour vérifier l'état de l'API."""
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    app.run(debug=True)
