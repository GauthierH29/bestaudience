import os
from datetime import datetime
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import pandas as pd
from dateutil.relativedelta import relativedelta
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler, RobustScaler
from sklearn.compose import make_column_transformer
<<<<<<< Updated upstream

=======
from sklearn.metrics.pairwise import cosine_similarity
>>>>>>> Stashed changes
import numpy as np
from collections import Counter
from bestaudience_py.ml_logic.recommend_sys import get_list_subcategories_unique, get_list_users_unique, calculate_user_item_matrix
from bestaudience_py.ml_logic.recommend_sys import get_top_similar_users, top_products_top_similar_users, top_products_user_selected, get_unique_products_for_users
from bestaudience_py.ml_logic.ml_kmeans_basic.model import find_optimal_k, fit_model, model_labels
from bestaudience_py.ml_logic.ml_kmeans_basic.preprocessor import future_data_processing, preprocess_data
from bestaudience_py.ml_logic.ml_kmeans_pca.model import find_optimal_threshold, transform_pca
from bestaudience_py.ml_logic.ml_kmeans_pca.preprocessor import features_engineering_PCA,groupby_client_PCA,preprocessing_for_PCA
from bestaudience_py.ml_logic.data import cleaning_data
from bestaudience_py.ml_logic.ml_kmeans_basic.registry import save_model
from bestaudience_py.params import MODEL_TYPE,MAX_K


<<<<<<< Updated upstream
=======

#current_directory = os.getcwd()
#parent_directory = os.path.dirname(current_directory)
#csv_path = os.path.join(parent_directory, 'raw_data', 'data_base_le_wagon.csv')
>>>>>>> Stashed changes
csv_path = os.path.join('raw_data', 'data_base_le_wagon.csv')
raw_data = pd.read_csv(csv_path,sep=';')

data = cleaning_data(raw_data)  # Complèter avec la fonction pour récupérer les données cleanées


if MODEL_TYPE == 'kmeans':
    data_gb = future_data_processing(data)
    data_scaled = preprocess_data(data_gb)
    optimal_n = find_optimal_k(data_scaled, MAX_K, MODEL_TYPE)
    model = fit_model(MODEL_TYPE, data_scaled, optimal_n=optimal_n)
    save_model(model,MODEL_TYPE)
    labels = model_labels(model)
    my_labels=dict()
    my_labels['label']=labels

    #print(labels)
    #ajout de la fonction label + df sans scaling

elif MODEL_TYPE == 'pca':
    data_cleaned = data
    data_groupby_cleaned = groupby_client_PCA(data_cleaned)
    new_data_groupby_cleaned = features_engineering_PCA(data_groupby_cleaned)
    data_preprocessed = preprocessing_for_PCA(new_data_groupby_cleaned)
    optimal_components = find_optimal_threshold(data_preprocessed, variance_threshold=0.90)
    model_pca = fit_model(MODEL_TYPE, data_preprocessed, optimal_n=None)
    transformed_data = transform_pca(model_pca, data_preprocessed, num_components=optimal_components)
    optimal_n = find_optimal_k(transformed_data, MAX_K, "kmeans")
    model = fit_model("kmeans", transformed_data, optimal_n=optimal_n)
    labels = model_labels(model)
    #print(labels)
    #ajout de la fonction label + df sans scaling

else:
    raise ValueError("Le choix du modèle est incorrect. Choisissez 'pca' ou 'kmeans'")

# Utiliser les résultats (labels) du modèle
<<<<<<< Updated upstream
def test():
    print("test")
=======




####  Recommandation product

#user_ids à définir
user_ids = ['CLT91838', 'CLT32918', 'CLT94868']

list_subcategories = get_list_subcategories_unique(data)
list_users_unique = get_list_users_unique(data)
user_item_matrix = calculate_user_item_matrix(data,list_users_unique,list_subcategories)
top_similar_users = get_top_similar_users(user_ids, user_item_matrix, list_users_unique, top_n=10)
top_products_similar_users = top_products_top_similar_users(user_ids, user_item_matrix, list_users_unique, list_subcategories,top_n_similar=10, top_n_products=5)
top_products_users_selected = top_products_user_selected(user_ids, user_item_matrix, list_users_unique, list_subcategories, top_n_products=3)
final_tab = get_unique_products_for_users(user_ids, user_item_matrix, list_users_unique, list_subcategories, top_n_similar=10, top_n_products=5)
print(final_tab)
>>>>>>> Stashed changes
