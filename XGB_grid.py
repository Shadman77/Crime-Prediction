from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn import model_selection
import pandas as pd
import pickle
from modules import grid_search


def perform_grid_search():
    df = pd.read_csv('data/smote.csv')

    # Seperate X and y
    X = df.drop(columns=["Primary Type"])
    y = df["Primary Type"]
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.3)  # 70:30

    # Set Parameters
    params = {
        "eta": [0.2, 0.3, 0.4],
        "max_depth": [6, 10, 14, 18, 22, 26, 30],
        "subsample": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    }

    grid_search.grid_search(params, XGBClassifier, X_train, y_train, "XGB")


if __name__ == "__main__":
    opt = int(input(
        "Enter 1 to perform grid search and 2 to get best parameters from results: "))

    if opt == 1:
        perform_grid_search()
    elif opt == 2:
        best_results, best_params = grid_search.get_best_params("XGB")
        for k in best_params:
            print()
            print("k = ", k)
            print("Result = ", best_results[k])
            print("Parameter = ", best_params[k])
