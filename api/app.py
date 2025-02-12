from flask import Flask, request, jsonify
import joblib
import pandas as pd
import os

# Initialiser l'application Flask
app = Flask(__name__)

def load_model():
    """Charge automatiquement le fichier .pkl dans le répertoire courant."""
    pkl_files = [f for f in os.listdir('.') if f.endswith('.pkl')]
    if not pkl_files:
        raise FileNotFoundError("Aucun fichier .pkl trouvé dans le répertoire courant.")
    # Charger le premier fichier .pkl trouvé
    model_filename = pkl_files[0]
    print(f"Chargement du modèle depuis {model_filename}")
    return joblib.load(model_filename)

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
