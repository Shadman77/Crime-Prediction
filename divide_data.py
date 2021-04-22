import pandas as pd
import pickle
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    best_k = 20
    df = pd.read_csv('data/smote.csv')

    # Seperate X and y
    X = df.drop(columns=["Primary Type"])
    y = df["Primary Type"]

    # Drop IUCR if it exists
    try:
        X = X.drop(columns=["IUCR"])
    except:
        pass

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=1/float(best_k))

    data = {
        "X_train": X_train,
        "X_val": X_val,
        "y_train": y_train,
        "y_val": y_val
    }

    path = "data/train_test.data"
    with open(path, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)



