import pandas as pd
from modules.grid_search import train_for_grid
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle


def get_best_k(results):
    best_res = 0
    best_k = 0

    for result in results:
        if results[result][0] > best_res:
            best_res = results[result][0]
            best_k = result

    return int(best_k)


def train_save(Model, param, name):
    KS = [10, 15, 20]
    # KS = [5, 10]

    # Load the dataset
    df = pd.read_csv('data/smote.csv')

    # Seperate X and y
    X = df.drop(columns=["Primary Type"])
    y = df["Primary Type"]

    # Get results
    results = train_for_grid(Model, param, KS, X, y)
    print(results)

    # Get the best value of K
    best_k = get_best_k(results)
    print("The best value of K is {}".format(best_k))

    # Train test split based on the best k value
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=1/float(best_k))

    # Train the final model
    model = Model(**param)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    print("Number of mislabeled points out of a total %d points : %d" %
          (X_val.shape[0], (y_val != y_pred).sum()))
    print("Accuracy is ", accuracy_score(y_val, y_pred))

    # Save the model
    path = "data/" + str(name) + ".model"
    with open(path, 'wb') as handle:
        pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)
