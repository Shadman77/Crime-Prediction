from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn import model_selection
import pandas as pd
import pickle, json
from modules import grid_search
from modules.train_best import train_save


def perform_grid_search():
    df = pd.read_csv('data/smote.csv')

    # Seperate X and y
    X = df.drop(columns=["Primary Type"])
    y = df["Primary Type"]
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.3)  # 70:30

    # Set Parameters
    params = {
        "var_smoothing": [1e-10, 1e-9, 1e-8]
    }

    grid_search.grid_search(
        params, GaussianNB, X_train, y_train, "GNB")


if __name__ == "__main__":
    opt = int(input(
        "Enter 1 to perform grid search and 2 to train with best parameter(s) from results: "))

    if opt == 1:
        perform_grid_search()
    elif opt == 2:
        best_results, best_params = grid_search.get_best_params("GNB")
        for k in best_params:
            print()
            print("k = ", k)
            print("Result = ", best_results[k])
            print("Parameter = ", best_params[k])
            param = json.loads(best_params[k])
            train_save(GaussianNB, param, "gnb_best", "Gaussian Naive Bayes")
