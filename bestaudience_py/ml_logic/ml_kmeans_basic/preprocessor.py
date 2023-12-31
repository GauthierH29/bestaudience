import pandas as pd

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from datetime import datetime
from dateutil.relativedelta import relativedelta
from bestaudience_py.ml_logic.data import load_data_to_bq
from bestaudience_py.params import GCP_PROJECT_ID
from bestaudience_py.ml_logic.recommend_sys.recommend_sys import rename_columns
from bestaudience_py.ml_logic.ml_kmeans_basic.registry import load_model

def calcul_anciennete(ma_date):
    if pd.isnull(ma_date):  # Handle missing values
        return 0

    try:
        diff = relativedelta(datetime.now(), ma_date)
        nb_mois = diff.years * 12 + diff.months
        return nb_mois
    except Exception as e:
        print(f"Error calculating ancienneté for date {ma_date}: {e}")
        return 0


def future_data_processing(data):
    # Regroupement des données par 'Client - ID' et sélection des premières valeurs
    data_gb = data.groupby('Client - ID').agg({'Client - Date de Naissance':'first', 'Client - Civilité':'first', 'Client - Mois de Création':'first', 'CA Produits HT':'sum', 'N Commandes':'first'})

    # Calcul de l'âge à partir de 'Client - Date de Naissance' et suppression de la colonne
    data_gb['age'] = 2023 - data_gb['Client - Date de Naissance'].dt.year
    data_gb.drop('Client - Date de Naissance', axis=1, inplace=True)

    # Calcul de l'ancienneté à partir de 'Client - Mois de Création' et suppression de la colonne
    data_gb['Ancienneté'] = data_gb['Client - Mois de Création'].apply(lambda x: calcul_anciennete(x))
    data_gb.drop('Client - Mois de Création', axis=1, inplace=True)

    data_gb.to_csv('notebook/kmeans_basics_analyse.csv', index=False)
    data_gb_rename=data_gb.reset_index()
    data_gb_rename = rename_columns(data_gb_rename)
    load_data_to_bq(data_gb_rename,GCP_PROJECT_ID,"data","kmeans_basics_analyse",truncate=True)

    return data_gb

def preprocess_data(data_gb):

    scaler = StandardScaler()

    data_scaled = data_gb.copy()
    data_scaled[['CA Produits HT', 'N Commandes', 'age', 'Ancienneté']] = scaler.fit_transform(data_gb[['CA Produits HT', 'N Commandes', 'age', 'Ancienneté']])

    ohe = OneHotEncoder(sparse_output=False)

    ohe.fit(data_gb[['Client - Civilité']])
    data_scaled[ohe.get_feature_names_out()] = ohe.transform(data_gb[['Client - Civilité']])
    data_scaled.drop(columns=["Client - Civilité"], inplace=True)

    return data_scaled
