
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import seaborn as sns


def find_optimal_threshold(data_scaled_pca, variance_threshold=0.90):
    pca = PCA()
    pca.fit(data_scaled_pca)
    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)

    for i, variance in enumerate(cumulative_variance):
        if variance >= variance_threshold:
            return i + 1  # Adding 1 because Python uses 0-based indexing

    # If no threshold satisfies the variance threshold, return the maximum threshold
    optimal_components = len(cumulative_variance)
    return optimal_components


def transform_pca(model_type, data_scaled_pca, num_components):

    transformed_data = model_type.transform(data_scaled_pca)
    transformed_data = pd.DataFrame(transformed_data[:, :num_components])
    return transformed_data


def find_optimal_k(data, max_k, model_type):
    inertia_values = []
    for k in range(1, max_k + 1):
        if model_type == 'kmeans':
            model = KMeans(n_clusters=k)
        else:
            raise ValueError("Le choix du modèle est incorrect. Choisissez 'kmeans'")

        model.fit(data)
        inertia_values.append(model.inertia_)

    diff = np.diff(inertia_values)
    elbow_index = np.argmax(diff) + 1
    optimal_n = elbow_index + 1
    return optimal_n


def fit_model(model_type, data, optimal_n):
    if model_type == 'pca':
        model = PCA()
    elif model_type == 'kmeans':
        model = KMeans(n_clusters=optimal_n, n_init=10)
    else:
        raise ValueError("Le choix du modèle est incorrect. Choisissez entre 'pca', 'kmeans'")

    model.fit(data)
    return model

def model_labels(model):
    return model.labels_
