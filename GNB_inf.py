import pickle

if __name__ == "__main__":

    # Load the saved model
    with open('data/gnb.model', 'rb') as handle:
        gnb = pickle.load(handle)