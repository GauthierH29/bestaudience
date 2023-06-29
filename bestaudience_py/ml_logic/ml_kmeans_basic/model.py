import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import seaborn as sns


def find_optimal_k(data_scaled, max_k=20):

    inertia_values = []
    for k in range(1, max_k + 1):
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(data_scaled)
        inertia_values.append(kmeans.inertia_)

    diff = np.diff(inertia_values)
    elbow_index = np.argmax(diff) + 1
    optimal_n = elbow_index + 1
    return optimal_n


optimal_n = find_optimal_k(data_scaled, max_k=20)

def create_kmeans(n_clusters=optimal_n, n_init=10):

    kmeans = KMeans(n_clusters=optimal_n, n_init=n_init)
    return kmeans

def fit_kmeans(kmeans, data_scaled):
    kmeans.fit(data_scaled)
    return kmeans

def get_labels(kmeans):
    return kmeans.labels_
