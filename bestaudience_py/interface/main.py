import os
from datetime import datetime
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import pandas as pd
from dateutil.relativedelta import relativedelta
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler, RobustScaler
from sklearn.compose import make_column_transformer
import numpy as np
from bestaudience_py.ml_logic.ml_kmeans_basic.model import find_optimal_k, fit_model, model_labels
from bestaudience_py.ml_logic.ml_kmeans_basic.preprocessor import future_data_processing, preprocess_data
from bestaudience_py.ml_logic.ml_kmeans_pca.model import find_optimal_threshold, transform_pca
from bestaudience_py.ml_logic.ml_kmeans_pca.preprocessor import features_engineering_PCA,groupby_client_PCA,preprocessing_for_PCA
from bestaudience_py.ml_logic.data import cleaning_data

#current_directory = os.getcwd()
#parent_directory = os.path.dirname(current_directory)
#csv_path = os.path.join(parent_directory, 'raw_data', 'data_base_le_wagon.csv')
csv_path = os.path.join('raw_data', 'data_base_le_wagon.csv')
raw_data = pd.read_csv(csv_path,sep=';')

data = cleaning_data(raw_data)  # Complèter avec la fonction pour récupérer les données cleanées

model_type = 'pca'  # Choix à faire pour les tests
max_k = 20  # Choix à faire sinon valeur par défaut

if model_type == 'kmeans':
    data_gb = future_data_processing(data)
    data_scaled = preprocess_data(data_gb)
    optimal_n = find_optimal_k(data_scaled, max_k, model_type)
    model = fit_model(model_type, data_scaled, optimal_n=optimal_n)
    labels = model_labels(model)
    print(labels)
    #ajout de la fonction label + df sans scaling

elif model_type == 'pca':
    data_cleaned = data
    data_groupby_cleaned = groupby_client_PCA(data_cleaned)
    new_data_groupby_cleaned = features_engineering_PCA(data_groupby_cleaned)
    data_preprocessed = preprocessing_for_PCA(new_data_groupby_cleaned)
    optimal_components = find_optimal_threshold(data_preprocessed, variance_threshold=0.90)
    model_pca = fit_model(model_type, data_preprocessed, optimal_n=None)
    transformed_data = transform_pca(model_pca, data_preprocessed, num_components=optimal_components)
    optimal_n = find_optimal_k(transformed_data, max_k, "kmeans")
    model = fit_model("kmeans", transformed_data, optimal_n=optimal_n)
    labels = model_labels(model)
    print(labels)
    #ajout de la fonction label + df sans scaling

else:
    raise ValueError("Le choix du modèle est incorrect. Choisissez 'pca' ou 'kmeans'")

# Utiliser les résultats (labels) du modèle
