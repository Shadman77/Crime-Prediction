
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn import model_selection
import pandas as pd
import pickle
from modules import grid_search

if __name__ == "__main__":
    df = pd.read_csv('data/smote.csv')

    #Seperate X and y
    X = df.drop(columns=["Primary Type"])
    y = df["Primary Type"]
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3) # 70:30

    # Set Parameters
    params = {
        # "eta": [0.3, 0.4],
        "max_depth": [6, 10],
        "subsample": [0.8, 1.0]
    }

    grid_search.grid_search(params, XGBClassifier, X_train, y_train, "XGB")