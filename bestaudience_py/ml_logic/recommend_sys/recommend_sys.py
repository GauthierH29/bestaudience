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

# création des variables :

list_subcategories = get_list_subcategories_unique()
list_users_unique = get_list_users_unique()

###Calcul de la matrice user_item_matrix

def calculate_user_item_matrix(df):
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

def get_top_similar_users(user_id, user_item_matrix, top_n=10):
    user_index = np.where(list_users_unique == user_id)[0][0]
    user_vector = user_item_matrix[user_index]

    similarities = cosine_similarity([user_vector], user_item_matrix)[0]
    #print(similarities)
    #print(similarities.shape)
    similar_users_indices = similarities.argsort()[::-1][1:top_n+1]

    similar_users = list_users_unique[similar_users_indices]
    return similar_users


### Récupération des top produits des top utilisateurs similaires

def top_produtcs_top_similar_users(top_similar_users, user_item_matrix, top_n_products=5):
    top_products = []
    for user_id in top_similar_users:
        user_index = np.where(list_users_unique == user_id)[0][0]
        user_vector = user_item_matrix[user_index]
        product_indices = np.argsort(user_vector)[::-1][:top_n_products]
        top_products.append(list_subcategories[product_indices])

    return list(np.unique(top_products))


### Récupération des top produits de l'utilisateur sélectionné

def top_products_user_selected(user_id,user_item_matrix,top_n_products=3):
    top_products = []
    user_index = np.where(list_users_unique == user_id)[0][0]
    user_vector = user_item_matrix[user_index]
    product_indices = np.argsort(user_vector)[::-1][:top_n_products]
    top_products.append(list_subcategories[product_indices])

    return list(np.unique(top_products))


### Comparaison des top produits utilisateurs similaires vs top produits utilisateur sélectionné

def compare_top_product_lists(user_id, user_item_matrix, top_n_similar=10, top_n_products=5):

    similar_users = get_top_similar_users(user_id, user_item_matrix, top_n=top_n_similar)
    top_products_similar_users = top_produtcs_top_similar_users(similar_users, user_item_matrix, top_n_products=top_n_products)
    top_products_user = top_products_user_selected(user_id, user_item_matrix, top_n_products=top_n_products)

    same_products = set(top_products_similar_users) & set(top_products_user)
    unique_products_similar_users = set(top_products_similar_users) - same_products
    unique_products_user_selected = set(top_products_user) - same_products

    return {
        "same_products": list(same_products),
        "unique_products_similar_users": list(unique_products_similar_users),
        "unique_products_user": list(unique_products_user_selected)
    }
