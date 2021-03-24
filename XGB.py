from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn import model_selection
import pandas as pd
import pickle

if __name__ == "__main__":
    df = pd.read_csv('data/smote.csv')

    #Seperate X and y
    X = df.drop(columns=["Primary Type"])
    y = df["Primary Type"]
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3) # 70:30

    # Define Model
    xgb = XGBClassifier(eta = 0.4)

    # Pexgborm 10 fold cross validation
    kfold = model_selection.KFold(n_splits=10)
    results = model_selection.cross_val_score(xgb, X_train, y_train, cv=kfold, verbose=2, n_jobs=2)
    print("Accuracy: %.3f%% (%.3f%%)" % (results.mean()*100.0, results.std()*100.0))

    # Train a new model one to save it
    xgb = XGBClassifier(eta = 0.4)
    xgb.fit(X_train, y_train)
    y_pred = xgb.predict(X_val)
    print("Number of mislabeled points out of a total %d points : %d" % (X_val.shape[0], (y_val != y_pred).sum()))
    print("Accuracy is ", accuracy_score(y_val, y_pred))

    # Save the model
    with open('data/xgb.model', 'wb') as handle:
        pickle.dump(xgb, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Save the validation set
    with open('data/xgb_example.pd', 'wb') as handle:
        pickle.dump(X_val.head(2), handle, protocol=pickle.HIGHEST_PROTOCOL)

    # To load the model
    # with open('filename.pickle', 'rb') as handle:
    #     gnb = pickle.load(handle)