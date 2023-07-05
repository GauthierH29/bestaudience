from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from bestaudience_py.ml_logic.ml_kmeans_basic.registry import load_model,load_model_from_bucket
from google.cloud import storage

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


@app.get('/kmean/Predict')
def kmean_predict(nb_k):
    #model = load_model('kmeans',nb_k)
    model=load_model_from_bucket('kmeans',nb_k)
    return {'labels':model.labels_.tolist()}

@app.get('/recommend_sys/Predict')
def recommend_sys(list_of_users,nb_voisins):

    return #renvoi un dictionnaire où la clé et un utilisateur
           #et la valeur une liste de tuple (nom du produit, occurence d'apparition)
           #ocurrence d'aparition correspond au nb de fois que le produit a été achété par les users le plus proches et que la cible n'a pas acheté

@app.get('/kmean_pca/Predict')
def kmean_pca_predict(nb_K):
    return #le df concat des labels avec le df cleaned
           #pour ensuite faire de la data viz sur l'app
