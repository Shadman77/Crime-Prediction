from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
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
        "C": [0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2]
    }

    grid_search.grid_search(
        params, SVC, X_train, y_train, "SVC")


if __name__ == "__main__":
    opt = int(input(
        "Enter 1 to perform grid search and 2 to train with best parameter(s) from results: "))

    if opt == 1:
        perform_grid_search()
    elif opt == 2:
        best_results, best_params = grid_search.get_best_params("SVC")
        for k in best_params:
            print()
            print("k = ", k)
            print("Result = ", best_results[k])
            print("Parameter = ", best_params[k])
            param = json.loads(best_params[k])
            train_save(SVC, param, "svc_best")
