import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import seaborn as sns
from sklearn.decomposition import PCA


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
