import pandas as pd
import os

# Chemin vers votre fichier CSV
# Assurez-vous que ce fichier est dans le même répertoire que ce script Python
CAR_DATA_LOCAL_PATH = 'vehicles_cleaned_for_training.csv'

def get_unique_manufacturers(file_path):
    """
    Charge un fichier CSV, extrait la colonne 'marque_nettoye',
    nettoie les données et renvoie une liste de constructeurs uniques et triés.
    """
    if not os.path.exists(file_path):
        print(f"Erreur : Le fichier '{file_path}' est introuvable. Veuillez vous assurer qu'il existe.")
        return []

    try:
        df_car_data = pd.read_csv(file_path)

        # Vérifier si la colonne 'marque_nettoye' existe
        if 'marque_nettoye' in df_car_data.columns:
            # Nettoyer les marques : convertir en string, mettre en majuscules, supprimer les espaces
            unique_manufacturers = df_car_data['marque_nettoye'].astype(str).str.upper().str.strip().unique().tolist()
            
            # Filtrer les valeurs potentiellement indésirables comme 'NAN', 'INCONNU' ou les chaînes vides
            # Note : Si 'AUTRE' est une catégorie valide dans vos données (suite à reduce_cardinality),
            # elle sera incluse ici.
            available_manufacturers = [m for m in unique_manufacturers if m and m not in ['NAN', 'INCONNU']] 
            
            # Trier la liste pour une meilleure lisibilité
            available_manufacturers.sort()
            
            return available_manufacturers
        else:
            print(f"Erreur : La colonne 'marque_nettoye' n'a pas été trouvée dans le fichier '{file_path}'.")
            return []
    except Exception as e:
        print(f"Une erreur est survenue lors de la lecture du fichier CSV ou du traitement des données : {e}")
        return []

if __name__ == "__main__":
    manufacturers = get_unique_manufacturers(CAR_DATA_LOCAL_PATH)
    
    if manufacturers:
        print("Constructeurs uniques disponibles :")
        for manufacturer in manufacturers:
            print(f"{manufacturer}")
    else:
        print("Aucun constructeur trouvé ou une erreur est survenue.")
