
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
