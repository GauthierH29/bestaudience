import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from bestaudience_py.ml_logic.data import load_data_to_bq
from bestaudience_py.params import GCP_PROJECT_ID
from bestaudience_py.ml_logic.data import cleaning_data, get_data_with_bq
from pathlib import Path

### Définir la liste des utilisateurs uniques
def get_list_users_unique(df):
    list_users_unique = np.unique(df['Client - ID'])
    return list_users_unique

### Définir la liste des sous catégories uniques
def get_list_subcategories_unique(df):
    list_subcategories = np.unique(df['Produit - Forme'])
    return list_subcategories

def remove_rows_with_slash(df):
    df = df[~df['Produit - Forme'].str.contains('/')]
    return df

###Calcul de la matrice user_item_matrix
def calculate_user_item_matrix(df:pd.DataFrame
                               ,list_users_unique
                               ,list_subcategories):
    #list_users_unique = np.unique(df['Client - ID'])
    #list_subcategories = np.unique(df['Produit - Forme'])
    user_item_matrix = np.zeros((len(list_users_unique), len(list_subcategories)))

    for index, row in df.iterrows():
        user = row['Client - ID']
        #print(user)
        subcategory = row['Produit - Forme']
        interaction = 1

        user_index = np.where(list_users_unique == user)[0][0]
        #print(np.where(list_users_unique == user))
        #print(user_index)
        subcategory_index = np.where(list_subcategories == subcategory)[0][0]

        user_item_matrix[user_index, subcategory_index] += interaction

    user_item =  pd.DataFrame(user_item_matrix, columns=list_subcategories, index=list_users_unique)
    #user_item.rename(columns={"/":"Frais de port"}, inplace=True)
    return user_item


### Récupération des top utilisateurs similaires
def get_top_similar_users(user_ids, user_item_matrix, list_users_unique, top_n=10):
    similar_users_dict = {}
    for user_id in user_ids:
        user_index = user_item_matrix.index.get_loc(user_id)
        user_vector = user_item_matrix.iloc[user_index]

        similarities = cosine_similarity([user_vector], user_item_matrix)[0]
        similar_users_indices = similarities.argsort()[::-1][1:top_n + 1]

        similar_users = list_users_unique[similar_users_indices]
        similar_users_dict[user_id] = list(similar_users)

    return similar_users_dict


### Récupération des top produits des top utilisateurs similaires

def top_products_top_similar_users(user_ids, user_item_matrix, list_users_unique, list_subcategories, top_n_similar=10, top_n_products=5):
    top_products_dict = {}
    for user_id in user_ids:
        user_index = user_item_matrix.index.get_loc(user_id)
        user_vector = user_item_matrix.iloc[user_index, :]

        similarities = cosine_similarity([user_vector], user_item_matrix)[0]
        similar_users_indices = similarities.argsort()[::-1][1:top_n_similar + 1]
        similar_users = list_users_unique[similar_users_indices]

        user_top_products = []
        for similar_user in similar_users:
            similar_user_index = user_item_matrix.index.get_loc(similar_user)
            similar_user_vector = user_item_matrix.iloc[similar_user_index, :]
            product_indices = np.argsort(similar_user_vector)[::-1][:top_n_products]
            top_products = list_subcategories[product_indices]
            user_top_products.extend(top_products)

        top_products_dict[user_id] = list(np.unique(user_top_products))

    return top_products_dict



### Récupération des top produits des utilisateurs sélectionnés

def top_products_user_selected(user_ids, user_item_matrix, list_users_unique, list_subcategories, top_n_products=3):
    top_products_dict = {}
    for user_id in user_ids:
        user_index = user_item_matrix.index.get_loc(user_id)
        user_vector = user_item_matrix.iloc[user_index, :]
        product_indices = np.argsort(user_vector)[::-1][:top_n_products]
        top_products = list_subcategories[product_indices]
        top_products_dict[user_id] = top_products

    return top_products_dict


### Comparaison des top produits utilisateurs similaires vs top produits utilisateur sélectionné

from collections import Counter

def get_unique_products_for_users(user_ids, user_item_matrix, list_users_unique, list_subcategories, top_n_similar=10, top_n_products=5):
    unique_products_dict = {}

    for user_id in user_ids:
        user_index = user_item_matrix.index.get_loc(user_id)
        user_vector = user_item_matrix.iloc[user_index]
        similarities = cosine_similarity([user_vector], user_item_matrix)[0]
        similar_users_indices = similarities.argsort()[::-1][1:top_n_similar+1]
        similar_users = list_users_unique[similar_users_indices]

        unique_products = []
        for similar_user in similar_users:
            similar_user_index = user_item_matrix.index.get_loc(similar_user)
            similar_user_vector = user_item_matrix.iloc[similar_user_index]
            product_indices = np.argsort(similar_user_vector)[::-1][:top_n_products]
            top_products = list_subcategories[product_indices]
            unique_products.extend(top_products)

        user_product_indices = np.argsort(user_vector)[::-1][:top_n_products]
        user_products = list_subcategories[user_product_indices]

        unique_products_counter = Counter(unique_products)
        unique_products_with_occurrence = [(product, count) for product, count in unique_products_counter.most_common() if product not in user_products]
        unique_products_dict[user_id] = unique_products_with_occurrence

    return unique_products_dict

def passer_index_en_colonne(df):
    user_item_matix_without_index = df.copy()
    user_item_matix_without_index.reset_index(inplace=True)
    return user_item_matix_without_index

def passer_colonne_en_index(df, name_c):
    new_user_matrix = df.copy()
    new_user_matrix.set_index(name_c, inplace=True)
    return new_user_matrix

### test
from unidecode import unidecode

def remove_accents(text):
    return unidecode(text)

def rename_columns(df):
    new_columns = []
    for column in df.columns:
        column = remove_accents(column)
        column = column.replace(' ', '_')
        column = column.replace('-', '_')
        column = column.lower()
        new_columns.append(column)
    df.columns = new_columns
    return df

def rename_columns_2(df):
    new_columns = []
    for column in df.columns:
        column = remove_accents(column)
        column = column.replace('-', '_')
        column = column.replace(' ', '')
        column = column.lower()
        new_columns.append(column)
    df.columns = new_columns
    return df

def creation_liste_from_string(string):
    test_list = string.split(',')
    test_list = [test_element.strip() for test_element in test_list]
    return test_list

if __name__=="__main__":

    #user_ids = ['CLT91838', 'CLT32918', 'CLT94868','CLT20208','CLT81083']
    user_ids = 'CLT91838'
    raw_data = pd.read_csv("raw_data/data_base_le_wagon.csv",sep=";")
    df = cleaning_data(raw_data)
    df = remove_rows_with_slash(df)

    user_item_matrix_table_name = "recommander.user_item_matrix"
    raw_data_user_matrix = Path("raw_data/user_matrix.csv")
    user_item_matrix_query = f"SELECT * FROM `{GCP_PROJECT_ID}.{user_item_matrix_table_name}`"

    list_users_unique = get_list_users_unique(df)
    list_subcategories = get_list_subcategories_unique(df)
    user_item_matrix = calculate_user_item_matrix(df, list_users_unique, list_subcategories)
    print(user_item_matrix)
    index_user_item_matrix = passer_index_en_colonne(user_item_matrix)
    index_user_item_matrix = rename_columns(index_user_item_matrix)
    print(index_user_item_matrix)
    load_data_to_bq(index_user_item_matrix,GCP_PROJECT_ID,"recommander","user_item_matrix",truncate=True)

    new_user_item_matrix = get_data_with_bq(GCP_PROJECT_ID,
        user_item_matrix_query
    )
    user_item_matrix = passer_colonne_en_index(new_user_item_matrix, "index")
    print(user_item_matrix)
    print(user_item_matrix.info())

    top_products_top_similar_user = top_products_top_similar_users(user_ids, user_item_matrix, list_users_unique, list_subcategories, top_n_similar=10, top_n_products=5)
    top_products_user_select = top_products_user_selected(user_ids, user_item_matrix, list_users_unique, list_subcategories, top_n_products=5)
    final_tab = get_unique_products_for_users(user_ids, user_item_matrix, list_users_unique, list_subcategories, top_n_similar=10, top_n_products=5)
    print(final_tab)
