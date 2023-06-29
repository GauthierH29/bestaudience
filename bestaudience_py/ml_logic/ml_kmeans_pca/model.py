
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import seaborn as sns


def fit_pca(data_scaled_pca):
    pca = PCA()
    pca.fit(data_scaled_pca)
    return pca


def find_optimal_threshold(data_scaled_pca, variance_threshold=0.90):
    pca = PCA()
    pca.fit(data_scaled_pca)
    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)

    for i, variance in enumerate(cumulative_variance):
        if variance >= variance_threshold:
            return i + 1  # Adding 1 because Python uses 0-based indexing

    # If no threshold satisfies the variance threshold, return the maximum threshold
    return len(cumulative_variance)

#data sera obtenu avec le get data + preprocessor
optimal_components = find_optimal_threshold(data, variance_threshold=0.90)

def transform_pca(pca, data_scaled_pca, num_components=optimal_components):

    transformed_data = pca.transform(data_scaled_pca)
    transformed_data = pd.DataFrame(transformed_data[:, :num_components])
    return transformed_data


def find_optimal_k(transformed_data, max_k=10):

    inertia_values = []
    for k in range(1, max_k + 1):
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(transformed_data)
        inertia_values.append(kmeans.inertia_)

    diff = np.diff(inertia_values)
    elbow_index = np.argmax(diff) + 1
    optimal_n = elbow_index + 1
    return optimal_n

def fit_kmeans(k, transformed_data):

    kmeans = KMeans(n_clusters=k)
    kmeans.fit(transformed_data)
    return kmeans

def get_labels(kmeans):
    return kmeans.labels_
