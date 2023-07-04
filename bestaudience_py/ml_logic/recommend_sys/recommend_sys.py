import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


### Définir la liste des utilisateurs uniques

def get_list_users_unique(df):
    list_users_unique = np.unique(df['Client - ID'])
    return list_users_unique

### Définir la liste des sous catégories uniques

def get_list_subcategories_unique(df):
    list_subcategories = np.unique(df['Produit - Forme'])
    return list_subcategories

# création des variables (à pousser sur le main) :

list_subcategories = get_list_subcategories_unique()
list_users_unique = get_list_users_unique()

###Calcul de la matrice user_item_matrix

def calculate_user_item_matrix(df,list_users_unique,list_subcategories):
    #list_users_unique = np.unique(df['Client - ID'])
    #list_subcategories = np.unique(df['Produit - Forme'])
    user_item_matrix = np.zeros((len(list_users_unique), len(list_subcategories)))

    for index, row in df.iterrows():
        user = row['Client - ID']
        #print(user)
        subcategory = row['Produit - Forme']
        #print(subcategory)
        interaction = 1

        user_index = np.where(list_users_unique == user)[0][0]
        #print(np.where(list_users_unique == user))
        #print(user_index)
        subcategory_index = np.where(list_subcategories == subcategory)[0][0]

        user_item_matrix[user_index, subcategory_index] += interaction

    return user_item_matrix


### Récupération des top utilisateurs similaires

def get_top_similar_users(user_ids, user_item_matrix, list_users_unique, top_n=10):
    similar_users_dict = {}
    for user_id in user_ids:
        user_index = np.where(list_users_unique == user_id)[0][0]
        user_vector = user_item_matrix[user_index]

        similarities = cosine_similarity([user_vector], user_item_matrix)[0]
        similar_users_indices = similarities.argsort()[::-1][1:top_n + 1]

        similar_users = list_users_unique[similar_users_indices]
        similar_users_dict[user_id] = list(similar_users)

    return similar_users_dict


### Récupération des top produits des top utilisateurs similaires

def top_products_top_similar_users(user_ids, user_item_matrix, list_users_unique, list_subcategories,top_n_similar=10, top_n_products=5):
    top_products_dict = {}
    for user_id in user_ids:
        user_index = np.where(list_users_unique == user_id)[0][0]
        user_vector = user_item_matrix[user_index]
        similarities = cosine_similarity([user_vector], user_item_matrix)[0]
        similar_users_indices = similarities.argsort()[::-1][1:top_n_similar+1]
        similar_users = list_users_unique[similar_users_indices]

        user_top_products = []
        for similar_user in similar_users:
            similar_user_index = np.where(list_users_unique == similar_user)[0][0]
            similar_user_vector = user_item_matrix[similar_user_index]
            product_indices = np.argsort(similar_user_vector)[::-1][:top_n_products]
            top_products = list_subcategories[product_indices]
            user_top_products.extend(top_products)

        top_products_dict[user_id] = list(np.unique(user_top_products))

    return top_products_dict


### Récupération des top produits des utilisateurs sélectionnés

def top_products_user_selected(user_ids, user_item_matrix, list_users_unique, list_subcategories, top_n_products=3):
    top_products_dict = {}
    for user_id in user_ids:
        user_index = np.where(list_users_unique == user_id)[0][0]
        user_vector = user_item_matrix[user_index]
        product_indices = np.argsort(user_vector)[::-1][:top_n_products]
        top_products = list_subcategories[product_indices]
        top_products_dict[user_id] = top_products

    return top_products_dict


### Comparaison des top produits utilisateurs similaires vs top produits utilisateur sélectionné

from collections import Counter

def get_unique_products_for_users(user_ids, user_item_matrix, list_users_unique, list_subcategories, top_n_similar=10, top_n_products=5):
    unique_products_dict = {}

    for user_id in user_ids:
        user_index = np.where(list_users_unique == user_id)[0][0]
        user_vector = user_item_matrix[user_index]
        similarities = cosine_similarity([user_vector], user_item_matrix)[0]
        similar_users_indices = similarities.argsort()[::-1][1:top_n_similar+1]
        similar_users = list_users_unique[similar_users_indices]

        unique_products = []
        for similar_user in similar_users:
            similar_user_index = np.where(list_users_unique == similar_user)[0][0]
            similar_user_vector = user_item_matrix[similar_user_index]
            product_indices = np.argsort(similar_user_vector)[::-1][:top_n_products]
            top_products = list_subcategories[product_indices]
            unique_products.extend(top_products)

        user_product_indices = np.argsort(user_vector)[::-1][:top_n_products]
        user_products = list_subcategories[user_product_indices]

        unique_products_counter = Counter(unique_products)
        unique_products_with_occurrence = [(product, count) for product, count in unique_products_counter.most_common() if product not in user_products]
        unique_products_dict[user_id] = unique_products_with_occurrence

    return unique_products_dict
