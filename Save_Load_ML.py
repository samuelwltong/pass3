import pickle

def save_ML(search, filename):
    pickle.dump(search, open(filename, 'wb'))
    print('Trained model saved.')

def load_ML(filename):
    loaded_model = pickle.load(open(filename, 'rb'))
    return loaded_model