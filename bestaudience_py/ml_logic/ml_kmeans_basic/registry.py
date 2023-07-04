import pickle

def save_model(model,model_name,):
    with open(f'models/{model_name}.pickle',mode="wb") as file:
        pickle.dump(model,file)



def load_model(model_name):
    path=f'models/{model_name}'
    return pickle.load(path)
