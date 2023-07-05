from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from bestaudience_py.ml_logic.ml_kmeans_basic.registry import load_model
from bestaudience_py.params import GCP_PROJECT_ID
from bestaudience_py.ml_logic.data import cleaning_data, get_data_with_bq, get_data_with_cache
from bestaudience_py.ml_logic.recommend_sys.recommend_sys import remove_rows_with_slash, get_list_users_unique, get_list_subcategories_unique, passer_colonne_en_index, creation_liste_from_string
from bestaudience_py.ml_logic.recommend_sys.recommend_sys import top_products_top_similar_users, top_products_user_selected, get_unique_products_for_users
from pathlib import Path

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Define a root `/` endpoint
@app.get('/')
def index():
    return {'ok': True}


@app.get('/KMEAN/Predict')
def KMEAN_predict(user):
    model = load_model(model_name)


@app.get('/Recommend/Predict')
def process_data(user_ids, top_n_similar, top_n_products):
    gcp_project = GCP_PROJECT_ID
    raw_data_table_name = "data.raw_data"
    user_item_matrix_table_name = "recommander.user_item_matrix"
    raw_data_query = f"SELECT * FROM `{gcp_project}.{raw_data_table_name}`"
    user_item_matrix_query = f"SELECT * FROM `{gcp_project}.{user_item_matrix_table_name}`"
    raw_data_cache_path = Path("raw_data/data_base_le_wagon.csv")
    raw_data_user_matrix = Path("raw_data/user_matrix.csv")


    raw_data = get_data_with_cache(gcp_project, raw_data_query, raw_data_cache_path, sep=";")

    df = cleaning_data(raw_data)
    df = remove_rows_with_slash(df)

    list_users_unique = get_list_users_unique(df)
    list_subcategories = get_list_subcategories_unique(df)

    new_user_item_matrix = get_data_with_bq(GCP_PROJECT_ID,
        user_item_matrix_query
    )
    user_item_matrix = passer_colonne_en_index(new_user_item_matrix, "index")
    #print(user_item_matrix.head())

    #top_products_top_similar_user = top_products_top_similar_users(user_ids, user_item_matrix, list_users_unique, list_subcategories, top_n_similar, top_n_products)
    #top_products_user_select = top_products_user_selected(user_ids, user_item_matrix, list_users_unique, list_subcategories, top_n_products)
    # fonction pour modifier user_ids
    user_ids = creation_liste_from_string(user_ids)
    final_tab = get_unique_products_for_users(user_ids, user_item_matrix, list_users_unique, list_subcategories, int(top_n_similar), int(top_n_products))

    return final_tab
