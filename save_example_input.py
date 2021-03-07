import pandas as pd
import pickle

if __name__ == "__main__":
    df = pd.read_csv('data/smote.csv')

    #Seperate X and y
    X = df.drop(columns=["Primary Type"])
    y = df["Primary Type"]

    #Take the first row
    X = X.iloc[0]


    # Save the model
    with open('data/example_input.pd', 'wb') as handle:
        pickle.dump(X, handle, protocol=pickle.HIGHEST_PROTOCOL)