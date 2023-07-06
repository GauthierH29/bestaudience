import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta
from sklearn.preprocessing import MinMaxScaler,RobustScaler,StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from bestaudience_py.ml_logic.data import load_data_to_bq
from bestaudience_py.params import GCP_PROJECT_ID
from bestaudience_py.ml_logic.recommend_sys.recommend_sys import rename_columns

#etapes restantes : importer data cleaned depuis get data dans data.py

def groupby_client_PCA(data_cleaned):
    data_groupby_cleaned = data_cleaned.copy()

    ohe=OneHotEncoder(sparse_output=False, drop='if_binary',handle_unknown = "ignore")

    #encodage de la feature, on transforme en gardant le nom des colonnes et on drop la colonne de base
    ohe.fit(data_cleaned[['Produit - Forme']])
    data_groupby_cleaned[ohe.get_feature_names_out()] = ohe.transform(data_cleaned[['Produit - Forme']])
    data_groupby_cleaned.drop(columns = ["Produit - Forme"], inplace = True)

    #Je crée un dict avec les features que je veux et l'opération à faire sur la colonne.

    my_dict={'Client - Date de Naissance':'first'
            ,'Client - Civilité':'first'
            ,'Client - Mois de Création':'first'
            ,'CA Produits HT':'sum'
            ,'N Commandes':'first'}

     #Je le fais avec une boucle for pour les features catégories que j'ai encodé. J'en enleve une car elle se nomme : "/"
    col_a_sum=data_groupby_cleaned.columns[16:].tolist()

    for i in col_a_sum:
        my_dict[i]='sum'

    data_groupby_cleaned=data_groupby_cleaned.groupby('Client - ID').agg(my_dict)

    return data_groupby_cleaned


def calcul_anciennete(ma_date):
    diff = relativedelta(datetime.now(), ma_date)
    nb_mois = diff.years*12 + diff.months
    return nb_mois

def features_engineering_PCA(data_groupby_cleaned):
    new_data_groupby_cleaned=data_groupby_cleaned.copy()
    #Fonction qui calcul l'ancienneté du client (par rapport à aujd)

    #Création de la variable age et drop la variable date de naissance
    new_data_groupby_cleaned['age']=2023 - new_data_groupby_cleaned['Client - Date de Naissance'].dt.year
    new_data_groupby_cleaned=new_data_groupby_cleaned.drop('Client - Date de Naissance',axis=1)

    #Création de la variable ancienneté (en mois) et drop de la date de création
    new_data_groupby_cleaned['Ancienneté'] = new_data_groupby_cleaned['Client - Mois de Création'].apply(lambda x: calcul_anciennete(x))
    new_data_groupby_cleaned=new_data_groupby_cleaned.drop('Client - Mois de Création',axis=1)

    new_data_groupby_cleaned.to_csv('notebook/kmeans_pca_analyse.csv', index=False)
    new_data_groupby_cleaned_rename=new_data_groupby_cleaned.reset_index()
    new_data_groupby_cleaned_rename = rename_columns(new_data_groupby_cleaned_rename)

    load_data_to_bq(new_data_groupby_cleaned_rename,GCP_PROJECT_ID,"data","kmeans_pca_analyse",truncate=True)

    return new_data_groupby_cleaned


def preprocessing_for_PCA(new_data_groupby_cleaned):
    #choix des scalers et encoders
    features_robust=['CA Produits HT','N Commandes']
    features_standard=['age']
    features_minmax=['Ancienneté']
    features_encoding=['Client - Civilité']
    features_num=features_minmax+features_robust+features_standard

    #scling des num features
    scalers = make_column_transformer(
    (RobustScaler(),features_robust),
    (StandardScaler(),features_standard),
    (MinMaxScaler(),features_minmax))

    #encode les cat features
    encoders = make_column_transformer(
    (OneHotEncoder(sparse_output=False, drop='if_binary',handle_unknown = "ignore"), features_encoding))

    preprocessor = make_column_transformer(
    (encoders,features_encoding),
    (scalers,features_num),
    remainder='passthrough')#garde les colonnes autres que celle scalées et encodées

    #get_features_names_out permet de garder les noms des colonnes, sinon elles sont rename par de schiffres
    data_preprocessed = pd.DataFrame(preprocessor.fit_transform(new_data_groupby_cleaned),columns=preprocessor.get_feature_names_out())

    return data_preprocessed.set_index(new_data_groupby_cleaned.index)
