import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

print("--- Démarrage du script d'entraînement du modèle ---")

# --- 1. Chargement des données nettoyées ---
input_cleaned_data_path = 'vehicles_cleaned_for_training.csv'
try:
    df = pd.read_csv(input_cleaned_data_path)
    print(f"Données chargées depuis '{input_cleaned_data_path}'. Dimensions : {df.shape}")
except FileNotFoundError:
    print(f"ERREUR : Le fichier '{input_cleaned_data_path}' est introuvable. Exécutez 'data_cleaner.py' d'abord.")
    exit()
except Exception as e:
    print(f"ERREUR lors du chargement des données : {e}")
    exit()

# Supprimer les lignes où la cible est NaN (devrait déjà être fait par le cleaner, mais sécurité)
df.dropna(subset=['prix_vente_nettoye'], inplace=True)
if df.empty:
    print("ERREUR : Le DataFrame est vide après suppression des NaN de la cible. Impossible d'entraîner.")
    exit()

# --- 2. Définition des caractéristiques (features) et de la cible (target) ---
target = 'prix_vente_nettoye'

# Caractéristiques numériques à mettre à l'échelle
numerical_features = [
    'annee_nettoye',
    'kilometrage_nettoye',
    'cylindres_nettoye',
    'age_vehicule',
    'condition_generale_nettoye_num'
]

# Caractéristiques catégorielles à encoder
categorical_features = [
    'region_localisation_nettoye',
    'marque_nettoye',
    'modele_nettoye',
    'type_carburant_nettoye',
    'transmission_nettoye',
    'roues_motrices_nettoye',
    'type_carrosserie_nettoye',
    'couleur_exterieure_nettoye'
]

# S'assurer que toutes les colonnes nécessaires existent dans le DataFrame
all_expected_features = numerical_features + categorical_features
for col in all_expected_features:
    if col not in df.columns:
        print(f"AVERTISSEMENT : La colonne attendue '{col}' est manquante dans le DataFrame nettoyé. Vérifiez 'data_cleaner.py'.")
        # Gérer les colonnes manquantes ici si nécessaire, par exemple en les ajoutant avec des NaN/valeurs par défaut
        if col in numerical_features:
            df[col] = np.nan
        else:
            df[col] = 'INCONNU'

# Imputation des valeurs manquantes pour les colonnes numériques AVANT la mise à l'échelle
# Utilisation de la médiane pour la robustesse aux outliers
for col in numerical_features:
    if df[col].isnull().any():
        median_val = df[col].median()
        df[col].fillna(median_val, inplace=True)
        print(f"Valeurs manquantes de '{col}' imputées avec la médiane : {median_val}")

# --- 3. Séparation des données en ensembles d'entraînement et de test ---
X = df[numerical_features + categorical_features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"\nDimensions des ensembles d'entraînement : X_train={X_train.shape}, y_train={y_train.shape}")
print(f"Dimensions des ensembles de test : X_test={X_test.shape}, y_test={y_test.shape}")

# --- 4. Prétraitement des caractéristiques : Encodage et Mise à l'échelle ---
print("\nÉtape 4/6: Prétraitement des caractéristiques (Encodage et Mise à l'échelle)...")

# Encodage One-Hot des caractéristiques catégorielles
ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
X_train_cat_encoded = ohe.fit_transform(X_train[categorical_features])
X_test_cat_encoded = ohe.transform(X_test[categorical_features])

# Créer des DataFrames à partir des résultats de l'encodage One-Hot
X_train_cat_df = pd.DataFrame(X_train_cat_encoded, columns=ohe.get_feature_names_out(categorical_features), index=X_train.index)
X_test_cat_df = pd.DataFrame(X_test_cat_encoded, columns=ohe.get_feature_names_out(categorical_features), index=X_test.index)

# Mise à l'échelle des caractéristiques numériques
scaler = StandardScaler()
X_train_num_scaled = scaler.fit_transform(X_train[numerical_features])
X_test_num_scaled = scaler.transform(X_test[numerical_features])

# Créer des DataFrames à partir des résultats de la mise à l'échelle
X_train_num_df = pd.DataFrame(X_train_num_scaled, columns=numerical_features, index=X_train.index)
X_test_num_df = pd.DataFrame(X_test_num_scaled, columns=numerical_features, index=X_test.index)

# Concaténer les caractéristiques numériques mises à l'échelle et les caractéristiques catégorielles encodées
X_train_processed = pd.concat([X_train_num_df, X_train_cat_df], axis=1)
X_test_processed = pd.concat([X_test_num_df, X_test_cat_df], axis=1)

print(f"Dimensions après encodage et mise à l'échelle : X_train_processed={X_train_processed.shape}")

# --- 5. Entraînement du modèle ---
print("\nÉtape 5/6: Entraînement du modèle (RandomForestRegressor)...")
# Vous pouvez changer ce modèle pour d'autres testés dans votre notebook (XGBoost, LightGBM, etc.)
model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train_processed, y_train)
print("Modèle entraîné avec succès.")

# --- 6. Évaluation du modèle ---
print("\nÉtape 6/6: Évaluation du modèle...")
y_pred = model.predict(X_test_processed)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"RMSE sur l'ensemble de test : {rmse:.2f}")
print(f"R² sur l'ensemble de test : {r2:.2f}")

# --- 7. Sauvegarde du modèle et des transformateurs ---
model_filename = 'modele_prediction_voiture_final.joblib'
ohe_filename = 'ohe_transformer.joblib'
scaler_filename = 'scaler_transformer.joblib'
numerical_features_filename = 'numerical_features.joblib'
categorical_features_filename = 'categorical_features.joblib'
final_model_features_filename = 'final_model_features.joblib' # Colonnes finales du modèle

try:
    joblib.dump(model, model_filename)
    joblib.dump(ohe, ohe_filename)
    joblib.dump(scaler, scaler_filename)
    joblib.dump(numerical_features, numerical_features_filename)
    joblib.dump(categorical_features, categorical_features_filename)
    joblib.dump(X_train_processed.columns.tolist(), final_model_features_filename) # Sauvegarde l'ordre et les noms des colonnes finales
    
    print(f"\nModèle enregistré sous : {model_filename}")
    print(f"Transformateur OneHotEncoder enregistré sous : {ohe_filename}")
    print(f"Transformateur StandardScaler enregistré sous : {scaler_filename}")
    print(f"Liste des caractéristiques numériques enregistrée sous : {numerical_features_filename}")
    print(f"Liste des caractéristiques catégorielles enregistrée sous : {categorical_features_filename}")
    print(f"Liste des colonnes finales du modèle enregistrée sous : {final_model_features_filename}")

except Exception as e:
    print(f"ERREUR lors de la sauvegarde des fichiers : {e}")

print("\n--- Entraînement du modèle terminé ---")