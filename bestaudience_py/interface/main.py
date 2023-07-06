import os
from datetime import datetime
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import pandas as pd
from dateutil.relativedelta import relativedelta
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler, RobustScaler
from sklearn.compose import make_column_transformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
#from bestaudience_py.ml_logic.recommend_sys import get_list_subcategories_unique, get_list_users_unique, calculate_user_item_matrix
#from bestaudience_py.ml_logic.recommend_sys import get_top_similar_users, top_products_top_similar_users, top_products_user_selected, get_unique_products_for_users
from bestaudience_py.ml_logic.ml_kmeans_basic.model import find_optimal_k, fit_model, model_labels
from bestaudience_py.ml_logic.ml_kmeans_basic.preprocessor import future_data_processing, preprocess_data
from bestaudience_py.ml_logic.ml_kmeans_pca.model import find_optimal_threshold, transform_pca
from bestaudience_py.ml_logic.ml_kmeans_pca.preprocessor import features_engineering_PCA,groupby_client_PCA,preprocessing_for_PCA
from bestaudience_py.ml_logic.data import cleaning_data
from bestaudience_py.ml_logic.ml_kmeans_basic.registry import save_model,save_model_to_bucket
from bestaudience_py.params import MODEL_TYPE,MAX_K




#current_directory = os.getcwd()
#parent_directory = os.path.dirname(current_directory)
#csv_path = os.path.join(parent_directory, 'raw_data', 'data_base_le_wagon.csv')

csv_path = os.path.join('raw_data', 'data_base_le_wagon.csv')
raw_data = pd.read_csv(csv_path,sep=';')

data = cleaning_data(raw_data)  # Complèter avec la fonction pour récupérer les données cleanées

def train_kmeans(model_type):
    if model_type == 'kmeans':
        data_gb = future_data_processing(data)
        data_scaled = preprocess_data(data_gb)
        optimal_n = find_optimal_k(data_scaled, MAX_K, model_type)
        model = fit_model(model_type, data_scaled, optimal_n=optimal_n)
        #save_model(model,MODEL_TYPE,optimal_n) #à supprimer
        for i in range(1,MAX_K+1):
            model = fit_model(model_type, data_scaled, optimal_n=i)
            save_model(model,model_type,i)#a supprimer
            save_model_to_bucket(model,model_type,i)
        labels = model_labels(model)#a supprimer
        my_labels=dict()#a supprimer
        my_labels['label']=labels#a supprimer



    elif model_type == 'pca':
        data_cleaned = data
        data_groupby_cleaned = groupby_client_PCA(data_cleaned)
        new_data_groupby_cleaned = features_engineering_PCA(data_groupby_cleaned)
        data_preprocessed = preprocessing_for_PCA(new_data_groupby_cleaned)
        optimal_components = find_optimal_threshold(data_preprocessed, variance_threshold=0.90)
        model_pca = fit_model(model_type, data_preprocessed, optimal_n=None)
        transformed_data = transform_pca(model_pca, data_preprocessed, num_components=optimal_components)
        optimal_n = find_optimal_k(transformed_data, MAX_K, "kmeans")
        model = fit_model("kmeans", transformed_data, optimal_n=optimal_n)
        for i in range(1,MAX_K+1):
            model = fit_model("kmeans", transformed_data, optimal_n=i)
            save_model(model,model_type,i)#a supprimer
            save_model_to_bucket(model,model_type,i)
        labels = model_labels(model)#a supprimer


    else:
        raise ValueError("Le choix du modèle est incorrect. Choisissez 'pca' ou 'kmeans'")

if __name__=="__main__":
    train_kmeans("pca")
    train_kmeans("kmeans")
