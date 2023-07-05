import os

MODEL_TYPE = os.environ.get("MODEL_TYPE")
MAX_K = int(os.environ.get('MAX_K'))
GCP_PROJECT_ID = os.environ.get('GCP_PROJECT_ID')

BUCKET_KMEAN=os.environ.get("BUCKET_KMEAN")
