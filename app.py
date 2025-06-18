import pandas as pd
import numpy as np
import re
import joblib
import os
from flask import Flask, request, render_template
import requests # Nouvelle importation nécessaire pour télécharger les fichiers

app = Flask(__name__)

# ==============================================================================
# SECTION GLOBALE : DÉFINITIONS ET MAPPAGES (IDENTIQUES À data_cleaner.py)
# ==============================================================================
col_mappings = {
    'price': 'prix_vente',
    'year': 'annee',
    'odometer': 'kilometrage',
    'cylinders': 'cylindres',
    'region': 'region_localisation',
    'manufacturer': 'marque',
    'model': 'modele',
    'condition': 'condition_generale',
    'fuel': 'type_carburant',
    'transmission': 'transmission',
    'drive': 'roues_motrices',
    'type': 'type_carrosserie',
    'paint_color': 'couleur_exterieure',
    'id': 'identifiant_annonce'
}

def reduce_cardinality(df, column_name, top_n=50, other_label='AUTRE'):
    """
    Réduit la cardinalité (nombre de valeurs uniques) d'une colonne catégorielle.
    Les 'top_n' catégories les plus fréquentes sont conservées, les autres sont regroupées.
    """
    if column_name not in df.columns:
        df[column_name] = other_label
        return df
    
    df[column_name] = df[column_name].astype(str)

    value_counts = df[column_name].value_counts()
    if len(value_counts) > top_n:
        top_categories = value_counts.index[:top_n]
        df[column_name] = df[column_name].apply(lambda x: x if x in top_categories else other_label)
    return df

# ==============================================================================
# FONCTION DE PRÉTRAITEMENT POUR LA PRÉDICTION (IDENTIQUE À predict.py)
# ==============================================================================
def preprocess_data_for_prediction(input_df_raw, ohe_transformer, scaler_transformer, numerical_features, categorical_features, final_model_features):
    """
    Prétraite les données brutes d'une nouvelle voiture pour la prédiction.
    Applique les mêmes transformations que celles utilisées lors de l'entraînement.
    """
    df_processed = input_df_raw.copy()

    # --- 1. Renommage et gestion des colonnes attendues (comme dans data_cleaner.py) ---
    for raw_name, clean_name in col_mappings.items():
        if raw_name in df_processed.columns:
            df_processed.rename(columns={raw_name: clean_name}, inplace=True)
        else:
            if clean_name in ['prix_vente', 'annee', 'kilometrage', 'cylindres']:
                df_processed[clean_name] = np.nan
            elif clean_name in ['region_localisation', 'marque', 'modele', 'condition_generale', 'type_carburant', 'transmission', 'roues_motrices', 'type_carrosserie', 'couleur_exterieure']:
                df_processed[clean_name] = 'INCONNU'
            else:
                df_processed[clean_name] = np.nan

    # --- 2. Nettoyage et conversion des types de données (comme dans data_cleaner.py) ---
    df_processed['prix_vente_nettoye'] = pd.to_numeric(df_processed['prix_vente'], errors='coerce')

    numeric_features_raw_temp = ['annee', 'kilometrage', 'cylindres']
    for col in numeric_features_raw_temp:
        df_processed[f'{col}_nettoye'] = pd.to_numeric(df_processed[col], errors='coerce')

    categorical_features_raw_temp = [
        'region_localisation', 'marque', 'modele', 'condition_generale',
        'type_carburant', 'transmission', 'roues_motrices', 'type_carrosserie',
        'couleur_exterieure'
    ]
    for col in categorical_features_raw_temp:
        df_processed[f'{col}_nettoye'] = df_processed[col].astype(str).str.upper().str.strip()
        df_processed[f'{col}_nettoye'].fillna('INCONNU', inplace=True)

    # --- 3. Réduction de Cardinalité (comme dans data_cleaner.py) ---
    df_processed = reduce_cardinality(df_processed, 'modele_nettoye', top_n=200)
    df_processed = reduce_cardinality(df_processed, 'marque_nettoye', top_n=50)
    df_processed = reduce_cardinality(df_processed, 'region_localisation_nettoye', top_n=20)
    df_processed = reduce_cardinality(df_processed, 'couleur_exterieure_nettoye', top_n=20)

    # --- 4. Feature Engineering (comme dans data_cleaner.py) ---
    current_year = pd.Timestamp.now().year
    df_processed['age_vehicule'] = current_year - df_processed['annee_nettoye']
    df_processed['age_vehicule'] = df_processed['age_vehicule'].apply(lambda x: x if x >= 0 else 0)
    df_processed['condition_generale_nettoye_num'] = pd.to_numeric(df_processed['condition_generale_nettoye'], errors='coerce')

    # --- 5. Imputation des valeurs manquantes pour les caractéristiques numériques ---
    for col in numerical_features:
        if col not in df_processed.columns or df_processed[col].isnull().any():
            df_processed[col].fillna(0, inplace=True) # Utilise 0 comme valeur par défaut

    # --- 6. Application des transformateurs entraînés ---

    X_num = df_processed[numerical_features]
    X_cat = df_processed[categorical_features]

    X_num_scaled = scaler_transformer.transform(X_num)
    X_num_df = pd.DataFrame(X_num_scaled, columns=numerical_features, index=df_processed.index)

    for col in categorical_features:
        if col not in X_cat.columns:
            X_cat[col] = 'INCONNU'

    X_cat_encoded = ohe_transformer.transform(X_cat)
    X_cat_df = pd.DataFrame(X_cat_encoded, columns=ohe_transformer.get_feature_names_out(categorical_features), index=df_processed.index)

    X_processed = pd.concat([X_num_df, X_cat_df], axis=1)
    
    missing_cols = set(final_model_features) - set(X_processed.columns)
    for c in missing_cols:
        X_processed[c] = 0 
    
    X_processed = X_processed[final_model_features]

    return X_processed

# ==============================================================================
# Chemins locaux et URLs de téléchargement (MODIFIEZ CES URLS !)
# ==============================================================================
# Ces URLs seront lues depuis les variables d'environnement de Railway.
# Assurez-vous de les configurer sur le tableau de bord Railway !
MODEL_DOWNLOAD_URL = os.environ.get('MODEL_URL')
OHE_DOWNLOAD_URL = os.environ.get('OHE_URL')
SCALER_DOWNLOAD_URL = os.environ.get('SCALER_URL')
NUM_FEATURES_DOWNLOAD_URL = os.environ.get('NUM_FEATURES_URL')
CAT_FEATURES_DOWNLOAD_URL = os.environ.get('CAT_FEATURES_URL')
FINAL_MODEL_FEATURES_DOWNLOAD_URL = os.environ.get('FINAL_MODEL_FEATURES_URL')

# Noms de fichiers qui seront créés localement dans le conteneur
MODEL_LOCAL_PATH = 'modele_prediction_voiture_final.joblib'
OHE_LOCAL_PATH = 'ohe_transformer.joblib'
SCALER_LOCAL_PATH = 'scaler_transformer.joblib'
NUM_FEATURES_LOCAL_PATH = 'numerical_features.joblib'
CAT_FEATURES_LOCAL_PATH = 'categorical_features.joblib'
FINAL_MODEL_FEATURES_LOCAL_PATH = 'final_model_features.joblib'

# Fonction pour télécharger un fichier s'il n'existe pas
def download_file_if_not_exists(url, local_filename):
    if not url:
        print(f"URL for {local_filename} is not set in environment variables. Cannot download.")
        return False
    if os.path.exists(local_filename):
        print(f"'{local_filename}' already exists locally. Skipping download.")
        return True
    print(f"Downloading '{local_filename}' from '{url}'...")
    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status() # Lève une erreur pour les mauvaises réponses (4xx ou 5xx)
            with open(local_filename, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        print(f"'{local_filename}' downloaded successfully.")
        return True
    except requests.exceptions.RequestException as e:
        print(f"ERREUR: Impossible de télécharger '{local_filename}' depuis '{url}': {e}")
        return False
    except Exception as e:
        print(f"Une erreur inattendue est survenue lors du téléchargement de '{local_filename}': {e}")
        return False

# Chargement du modèle et des transformateurs
model = None
ohe = None
scaler = None
numerical_features_list = None
categorical_features_list = None
final_model_features_list = None

# Cette fonction sera appelée au démarrage de l'application Flask
def load_all_model_resources():
    global model, ohe, scaler, numerical_features_list, categorical_features_list, final_model_features_list

    print("\nAttempting to download and load model resources...")
    
    # Tenter de télécharger tous les fichiers nécessaires
    success = True
    success &= download_file_if_not_exists(MODEL_DOWNLOAD_URL, MODEL_LOCAL_PATH)
    success &= download_file_if_not_exists(OHE_DOWNLOAD_URL, OHE_LOCAL_PATH)
    success &= download_file_if_not_exists(SCALER_DOWNLOAD_URL, SCALER_LOCAL_PATH)
    success &= download_file_if_not_exists(NUM_FEATURES_DOWNLOAD_URL, NUM_FEATURES_LOCAL_PATH)
    success &= download_file_if_not_exists(CAT_FEATURES_DOWNLOAD_URL, CAT_FEATURES_LOCAL_PATH)
    success &= download_file_if_not_exists(FINAL_MODEL_FEATURES_DOWNLOAD_URL, FINAL_MODEL_FEATURES_LOCAL_PATH)

    if not success:
        print("Erreur : Au moins un fichier n'a pas pu être téléchargé. Le modèle ne sera pas initialisé.")
        return False

    try:
        # Charger les fichiers joblib depuis les chemins locaux
        model = joblib.load(MODEL_LOCAL_PATH)
        ohe = joblib.load(OHE_LOCAL_PATH)
        scaler = joblib.load(SCALER_LOCAL_PATH)
        numerical_features_list = joblib.load(NUM_FEATURES_LOCAL_PATH)
        categorical_features_list = joblib.load(CAT_FEATURES_LOCAL_PATH)
        final_model_features_list = joblib.load(FINAL_MODEL_FEATURES_LOCAL_PATH)
        print("Modèle et transformateurs chargés avec succès depuis le stockage local du conteneur.")
        return True
    except FileNotFoundError as e:
        print(f"ERREUR : Un fichier du modèle ou des transformateurs est introuvable localement (même après tentative de téléchargement) : {e}")
        return False
    except Exception as e:
        print(f"ERREUR : Impossible de charger les composants du modèle depuis le disque. Détails : {e}")
        import traceback
        traceback.print_exc()
        return False

# Appel de la fonction de chargement des ressources au démarrage de l'application
# Gunicorn va exécuter ce code une seule fois au démarrage du processus de l'application
load_all_model_resources()

# ==============================================================================
# ROUTES DE L'APPLICATION FLASK (Identiques au code précédent)
# ==============================================================================

@app.route('/')
def home():
    """Affiche la page d'accueil avec le formulaire de prédiction."""
    current_year = pd.Timestamp.now().year
    return render_template('index.html', current_year=current_year)

@app.route('/predict', methods=['POST'])
def predict():
    """Gère la soumission du formulaire et renvoie la prédiction."""
    # Vérifier si tous les composants du modèle ont été chargés avec succès
    if model is None or ohe is None or scaler is None or numerical_features_list is None or categorical_features_list is None or final_model_features_list is None:
        return render_template('index.html', prediction_text="Erreur : Les composants du modèle n'ont pas pu être chargés au démarrage. La prédiction est impossible.",
                               current_year=pd.Timestamp.now().year)

    try:
        # Récupérer les données du formulaire
        form_data = {
            'year': request.form.get('year'), 'manufacturer': request.form.get('manufacturer'),
            'model': request.form.get('model'), 'odometer': request.form.get('odometer'),
            'cylinders': request.form.get('cylinders'), 'region': request.form.get('region'),
            'condition': request.form.get('condition'), 'fuel': request.form.get('fuel'),
            'transmission': request.form.get('transmission'), 'drive': request.form.get('drive'),
            'type': request.form.get('type'), 'paint_color': request.form.get('paint_color')
        }

        # Convertir les valeurs numériques du formulaire en type approprié
        for key in ['year', 'odometer', 'cylinders']:
            if form_data.get(key):
                try:
                    form_data[key] = float(form_data[key])
                except ValueError:
                    form_data[key] = np.nan

        # Créer un DataFrame à partir des données du formulaire (brutes)
        input_df_raw = pd.DataFrame([form_data])

        # Prétraiter les données de la nouvelle voiture
        processed_input_df = preprocess_data_for_prediction(
            input_df_raw, ohe, scaler, numerical_features_list, categorical_features_list, final_model_features_list
        )
        
        # Faire la prédiction
        predicted_price = model.predict(processed_input_df)[0]

        return render_template('index.html', prediction_text=f"Le prix de vente prédit est : {predicted_price:,.2f} AED",
                               current_year=pd.Timestamp.now().year)

    except Exception as e:
        print(f"Erreur lors de la prédiction : {e}")
        import traceback
        traceback.print_exc()
        return render_template('index.html', prediction_text=f"Une erreur inattendue est survenue lors de la prédiction : {e}. Vérifiez les valeurs saisies.",
                               current_year=pd.Timestamp.now().year)

if __name__ == '__main__':
    # Pour le développement local, cette partie sera exécutée.
    # En production avec Gunicorn (via Dockerfile CMD), c'est Gunicorn qui démarre l'application.
    port = int(os.environ.get('PORT', 5000)) # Utilise 5000 par défaut pour local, Railway injecte PORT
    app.run(debug=True, host='0.0.0.0', port=port)