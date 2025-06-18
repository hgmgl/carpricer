import pandas as pd
import numpy as np
import re

# ==============================================================================
# SECTION GLOBALE : DÉFINITIONS ET MAPPAGES
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
        print(f"AVERTISSEMENT: Colonne '{column_name}' non trouvée pour réduction de cardinalité. Création avec label par défaut.")
        df[column_name] = other_label
        return df

    df[column_name] = df[column_name].astype(str) # Assurez-vous que la colonne est de type string

    value_counts = df[column_name].value_counts()
    if len(value_counts) > top_n:
        top_categories = value_counts.index[:top_n]
        df[column_name] = df[column_name].apply(lambda x: x if x in top_categories else other_label)
    return df

def clean_and_preprocess_data(input_csv_path, output_csv_path='vehicles_cleaned_for_training.csv'):
    """
    Charge les données, effectue le nettoyage et le prétraitement, puis sauvegarde le DataFrame nettoyé.
    """
    print(f"--- Démarrage du nettoyage des données depuis '{input_csv_path}' ---")

    try:
        df = pd.read_csv(input_csv_path)
        print(f"Fichier '{input_csv_path}' chargé avec succès. Dimensions initiales : {df.shape}")
    except FileNotFoundError:
        print(f"ERREUR : Le fichier '{input_csv_path}' est introuvable. Assurez-vous qu'il est dans le même répertoire.")
        return None
    except Exception as e:
        print(f"ERREUR lors du chargement du fichier '{input_csv_path}' : {e}")
        return None

    df_processed = df.copy()

    # --- 1. Renommage et gestion des colonnes attendues ---
    print("\nÉtape 1/5: Renommage des colonnes et gestion des manquantes...")
    for raw_name, clean_name in col_mappings.items():
        if raw_name in df_processed.columns:
            df_processed.rename(columns={raw_name: clean_name}, inplace=True)
        else:
            print(f"AVERTISSEMENT : Colonne '{raw_name}' manquante. Ajout d'une colonne par défaut '{clean_name}'.")
            if clean_name in ['prix_vente', 'annee', 'kilometrage', 'cylindres']:
                df_processed[clean_name] = np.nan
            elif clean_name in ['region_localisation', 'marque', 'modele', 'condition_generale', 'type_carburant', 'transmission', 'roues_motrices', 'type_carrosserie', 'couleur_exterieure']:
                df_processed[clean_name] = 'INCONNU'
            else:
                df_processed[clean_name] = np.nan

    # --- 2. Nettoyage et conversion des types de données ---
    print("\nÉtape 2/5: Nettoyage et conversion des types de données...")
    
    # Cible : prix_vente
    df_processed['prix_vente_nettoye'] = pd.to_numeric(df_processed['prix_vente'], errors='coerce')

    # Caractéristiques numériques
    numeric_features_raw = ['annee', 'kilometrage', 'cylindres']
    for col in numeric_features_raw:
        if col in df_processed.columns:
            df_processed[f'{col}_nettoye'] = pd.to_numeric(df_processed[col], errors='coerce')
        else:
            df_processed[f'{col}_nettoye'] = np.nan

    # Caractéristiques catégorielles
    categorical_features_raw = [
        'region_localisation', 'marque', 'modele', 'condition_generale',
        'type_carburant', 'transmission', 'roues_motrices', 'type_carrosserie',
        'couleur_exterieure'
    ]
    for col in categorical_features_raw:
        if col in df_processed.columns:
            df_processed[f'{col}_nettoye'] = df_processed[col].astype(str).str.upper().str.strip()
        else:
            df_processed[f'{col}_nettoye'] = 'INCONNU'
        df_processed[f'{col}_nettoye'].fillna('INCONNU', inplace=True)

    # --- 3. Réduction de Cardinalité ---
    print("\nÉtape 3/5: Application de la réduction de cardinalité...")
    df_processed = reduce_cardinality(df_processed, 'modele_nettoye', top_n=200)
    df_processed = reduce_cardinality(df_processed, 'marque_nettoye', top_n=50)
    df_processed = reduce_cardinality(df_processed, 'region_localisation_nettoye', top_n=20)
    df_processed = reduce_cardinality(df_processed, 'couleur_exterieure_nettoye', top_n=20)

    # --- 4. Feature Engineering (Création de nouvelles caractéristiques) ---
    print("\nÉtape 4/5: Création de nouvelles caractéristiques...")
    current_year = pd.Timestamp.now().year
    df_processed['age_vehicule'] = current_year - df_processed['annee_nettoye']
    df_processed['age_vehicule'] = df_processed['age_vehicule'].apply(lambda x: x if x >= 0 else 0)

    df_processed['condition_generale_nettoye_num'] = pd.to_numeric(df_processed['condition_generale_nettoye'], errors='coerce')

    # --- 5. Gestion des valeurs manquantes critiques et des filtres ---
    print("\nÉtape 5/5: Gestion des valeurs manquantes critiques et application des filtres...")

    initial_rows_count = df_processed.shape[0]
    
    # Colonnes essentielles pour le modèle après prétraitement initial
    # Note: Toutes ces colonnes seront produites par les étapes précédentes
    # S'il y a des NaN, ils seront gérés par imputation plus tard (dans training_script.py ou predict.py)
    essential_final_features_for_cleaner_output = [
        'prix_vente_nettoye', # La cible
        'annee_nettoye',
        'kilometrage_nettoye',
        'cylindres_nettoye',
        'age_vehicule',
        'condition_generale_nettoye_num',
        'region_localisation_nettoye',
        'marque_nettoye',
        'modele_nettoye',
        'type_carburant_nettoye',
        'transmission_nettoye',
        'roues_motrices_nettoye',
        'type_carrosserie_nettoye',
        'couleur_exterieure_nettoye',
        'identifiant_annonce' # Garder l'ID si présent, utile pour le suivi
    ]

    # Filtrer les colonnes qui existent réellement dans df_processed
    final_columns_to_keep = [col for col in essential_final_features_for_cleaner_output if col in df_processed.columns]
    df_final = df_processed[final_columns_to_keep].copy()

    # Nettoyage des lignes avec valeurs manquantes pour la cible
    rows_before_target_drop = df_final.shape[0]
    df_final.dropna(subset=['prix_vente_nettoye'], inplace=True)
    print(f"Lignes supprimées car prix_vente_nettoye était manquant : {rows_before_target_drop - df_final.shape[0]}")
    
    if df_final.empty:
        print("ERREUR : Le DataFrame est vide après la suppression des lignes sans prix. Vérifiez votre fichier source.")
        return None

    # Application des filtres sur les plages de valeurs
    rows_before_km_filter = df_final.shape[0]
    df_final = df_final[df_final['kilometrage_nettoye'] >= 500]
    print(f"Lignes supprimées car kilométrage < 500 : {rows_before_km_filter - df_final.shape[0]}")

    rows_before_price_filter = df_final.shape[0]
    df_final = df_final[df_final['prix_vente_nettoye'] >= 0]
    print(f"Lignes supprimées car prix de vente < 0 : {rows_before_price_filter - df_final.shape[0]}")

    if df_final.empty:
        print("ERREUR : Le DataFrame est vide après l'application des filtres. Très peu de données valides.")
        return None

    print("\nFiltrage des valeurs aberrantes pour les colonnes numériques clés (1% et 99% centile)...")
    numeric_cols_for_outlier_filtering = [
        'prix_vente_nettoye', 'kilometrage_nettoye', 'cylindres_nettoye',
        'age_vehicule', 'condition_generale_nettoye_num'
    ]
    current_rows_count = df_final.shape[0]
    for col in numeric_cols_for_outlier_filtering:
        if col in df_final.columns and pd.api.types.is_numeric_dtype(df_final[col]):
            Q1 = df_final[col].quantile(0.01)
            Q3 = df_final[col].quantile(0.99)
            if Q1 < Q3:
                initial_rows_col_filter = df_final.shape[0]
                df_final = df_final[(df_final[col] >= Q1) & (df_final[col] <= Q3)]
                print(f"- '{col}': {initial_rows_col_filter - df_final.shape[0]} lignes supprimées pour les outliers.")
            else:
                print(f"AVERTISSEMENT : La colonne '{col}' a des valeurs trop uniformes (Q1 >= Q3). Pas de filtrage des outliers pour cette colonne.")
        else:
            print(f"AVERTISSEMENT : Colonne '{col}' non trouvée ou non numérique pour le filtrage des outliers.")
    print(f"Total lignes supprimées par filtrage des outliers : {current_rows_count - df_final.shape[0]}")

    if df_final.empty:
        print("ERREUR : Le DataFrame est vide après le filtrage des valeurs aberrantes.")
        return None
            
    print(f"\nDimensions du DataFrame nettoyé final : {df_final.shape}")
    print("\nAperçu des 5 premières lignes du DataFrame nettoyé :")
    print(df_final.head())
    print("\nInformations sur le DataFrame nettoyé :")
    df_final.info()

    try:
        df_final.to_csv(output_csv_path, index=False)
        print(f"\nDonnées nettoyées sauvegardées dans '{output_csv_path}'")
    except Exception as e:
        print(f"ERREUR lors de la sauvegarde du fichier nettoyé : {e}")

    print("\n--- Nettoyage des données terminé ---")
    return df_final

# ==============================================================================
# SECTION PRINCIPALE : EXÉCUTION DU NETTOYEUR
# ==============================================================================
if __name__ == "__main__":
    input_file = 'vehicles.csv' # Assurez-vous que votre fichier de données est ici
    output_file = 'vehicles_cleaned_for_training.csv'
    
    clean_and_preprocess_data(input_file, output_file) 