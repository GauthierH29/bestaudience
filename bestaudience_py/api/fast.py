from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from bestaudience_py.ml_logic.ml_kmeans_basic.registry import load_model

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
