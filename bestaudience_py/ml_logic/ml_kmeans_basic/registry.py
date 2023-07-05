import pickle
from google.cloud import storage
from bestaudience_py.params import MODEL_TYPE,MAX_K,BUCKET_KMEAN
import io

def save_model(model,model_name,nb_k):
    with open(f'models/{model_name}{nb_k}.pickle',mode="wb") as file:
        pickle.dump(model,file)

def load_model(model_name,nb_k):
    file_path=f'models/{model_name}{nb_k}.pickle'
    with open(file_path, 'rb') as file:
        model = pickle.load(file)
    return model

from google.cloud import storage

def save_model_to_bucket(model, model_name, nb_k):
    # Créer une instance du client Google Cloud Storage
    client = storage.Client()

    # Accéder au bucket spécifié
    bucket = client.bucket(BUCKET_KMEAN)

    # Définir le nom du fichier dans le bucket
    file_name = f"{model_name}{nb_k}.pickle"

    # Créer un objet blob dans le bucket
    blob = bucket.blob(file_name)

    # Enregistrer le modèle dans l'objet blob
    with blob.open(mode="wb") as file:
        pickle.dump(model, file)


def load_model_from_bucket(model_name, nb_k):
    storage_client = storage.Client()
    bucket = storage_client.bucket(BUCKET_KMEAN)
    blob = bucket.blob(f'{model_name}{nb_k}.pickle')

    # Télécharger le contenu du blob dans un flux de mémoire
    content_bytes = io.BytesIO()
    blob.download_to_file(content_bytes)
    content_bytes.seek(0)  # Revenir au début du flux

    # Charger le modèle depuis le flux de mémoire
    model = pickle.load(content_bytes)

    return model
