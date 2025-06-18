import pandas as pd
import numpy as np
import re
import joblib
import os

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
# FONCTION DE PRÉTRAITEMENT POUR LA PRÉDICTION (UTILISE LES TRANSFORMERS SAUVEGARDÉS)
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
    df_processed['prix_vente_nettoye'] = pd.to_numeric(df_processed['prix_vente'], errors='coerce') # Garder pour consistance, même si non utilisée directement pour X

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
    # Utilisation de la médiane entraînée par le scaler (qui est implicite dans scaler_transformer.mean_)
    # Ou une médiane générique si le scaler n'est pas censé gérer les NaN.
    # Pour des données de prédiction unitaires, il est plus simple d'imputer avant le scaler.
    for col in numerical_features:
        if col not in df_processed.columns or df_processed[col].isnull().any():
            # Si la colonne est entièrement NaN (nouvelle input), ou contient des NaN,
            # utilisez une valeur par défaut raisonnable (e.g., 0 ou médiane si connue)
            # Ici, pour la prédiction, on impute avec la moyenne/médiane des données d'entraînement si elle n'est pas NaN
            # ou avec 0 si le modèle gère les 0 pour les valeurs manquantes.
            # Pour l robustesse, utilisons la valeur 0 pour les Nans qui n'auraient pas été remplis
            df_processed[col].fillna(0, inplace=True) 

    # --- 6. Application des transformateurs entraînés ---

    # Sélection des colonnes numériques et catégorielles pour la transformation
    X_num = df_processed[numerical_features]
    X_cat = df_processed[categorical_features]

    # Mise à l'échelle des caractéristiques numériques
    X_num_scaled = scaler_transformer.transform(X_num)
    X_num_df = pd.DataFrame(X_num_scaled, columns=numerical_features, index=df_processed.index)

    # Encodage One-Hot des caractéristiques catégorielles
    # Gérer les colonnes manquantes dans l'input_df_raw avant d'appliquer OneHotEncoder
    # S'assurer que toutes les categorical_features sont présentes dans X_cat avant transform
    for col in categorical_features:
        if col not in X_cat.columns:
            X_cat[col] = 'INCONNU' # Ajoutez la colonne avec la valeur par défaut

    X_cat_encoded = ohe_transformer.transform(X_cat)
    X_cat_df = pd.DataFrame(X_cat_encoded, columns=ohe_transformer.get_feature_names_out(categorical_features), index=df_processed.index)

    # Concaténation des caractéristiques traitées
    X_processed = pd.concat([X_num_df, X_cat_df], axis=1)
    
    # Réaligner les colonnes avec celles utilisées lors de l'entraînement du modèle
    # Ceci est CRUCIAL pour que le modèle reçoive les colonnes dans le bon ordre et avec les bons noms
    missing_cols = set(final_model_features) - set(X_processed.columns)
    for c in missing_cols:
        X_processed[c] = 0 # Ajouter les colonnes manquantes avec des zéros (pour OneHotEncoder, représente l'absence)
    
    # Assurez-vous d'avoir uniquement les colonnes du modèle final et dans le bon ordre
    X_processed = X_processed[final_model_features]

    return X_processed

# ==============================================================================
# LOGIQUE PRINCIPALE DE PRÉDICTION
# ==============================================================================
print("--- Script de prédiction de prix de voiture ---")

# Chemins des fichiers sauvegardés
MODEL_PATH = 'modele_prediction_voiture_final.joblib'
OHE_PATH = 'ohe_transformer.joblib'
SCALER_PATH = 'scaler_transformer.joblib'
NUM_FEATURES_PATH = 'numerical_features.joblib'
CAT_FEATURES_PATH = 'categorical_features.joblib'
FINAL_MODEL_FEATURES_PATH = 'final_model_features.joblib'


# Vérifier l'existence des fichiers
required_files = [MODEL_PATH, OHE_PATH, SCALER_PATH, NUM_FEATURES_PATH, CAT_FEATURES_PATH, FINAL_MODEL_FEATURES_PATH]
for f_path in required_files:
    if not os.path.exists(f_path):
        print(f"ERREUR FATALE : Le fichier '{f_path}' est introuvable.")
        print("Veuillez d'abord exécuter 'training_script.py' pour générer ces fichiers.")
        exit()

try:
    # 1. Charger le modèle et les transformateurs
    model = joblib.load(MODEL_PATH)
    ohe = joblib.load(OHE_PATH)
    scaler = joblib.load(SCALER_PATH)
    numerical_features = joblib.load(NUM_FEATURES_PATH)
    categorical_features = joblib.load(CAT_FEATURES_PATH)
    final_model_features = joblib.load(FINAL_MODEL_FEATURES_PATH) # Les noms de colonnes attendus par le modèle

    print("Modèle et transformateurs chargés avec succès.")

    # 2. Définir les caractéristiques de la NOUVELLE voiture à prédire
    # Utilisez les noms de colonnes BRUTS ORIGINAUX de votre fichier CSV (ex: 'year', 'manufacturer').
    # Renseignez les informations que vous connaissez. Les champs manquants seront gérés.
    new_car_data_raw = pd.DataFrame([{
        "year": 2020,
        "manufacturer": "HONDA",
        "model": "CIVIC",
        "odometer": 80000,
        "cylinders": 4,
        "region": "dubai",
        "condition": "excellent",
        "fuel": "GAS",
        "transmission": "automatic",
        "drive": "fwd",
        "type": "sedan",
        "paint_color": "silver"
    }])

    print("\nDonnées de la nouvelle voiture pour la prédiction (brutes, avant prétraitement) :")
    print(new_car_data_raw)

    # 3. Prétraiter les données de la nouvelle voiture
    processed_input_df = preprocess_data_for_prediction(
        new_car_data_raw, ohe, scaler, numerical_features, categorical_features, final_model_features
    )

    print("\nDonnées après prétraitement (prêtes pour le modèle) :")
    print(processed_input_df.head())
    print(f"Dimensions des données prétraitées : {processed_input_df.shape}")

    # 4. Faire la prédiction
    predicted_price = model.predict(processed_input_df)[0]

    print(f"\nLe prix de vente prédit pour cette voiture est : {predicted_price:,.2f} usd")

except Exception as e:
    print(f"Une erreur est survenue lors de la prédiction : {e}")
    import traceback
    traceback.print_exc() # Pour afficher la trace complète de l'erreur

print("\n--- Fin du script de prédiction ---")