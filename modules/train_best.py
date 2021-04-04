import pandas as pd
from modules.grid_search import train_for_grid
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns


def get_best_k(results):
    best_res = 0
    best_k = 0

    for result in results:
        if results[result][0] > best_res:
            best_res = results[result][0]
            best_k = result

    return int(best_k)


def train_save(Model, param, name, title):
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

    # Generating classification report
    print('Classification report of ' + title + ' classifier:')
    print(classification_report(y_val, y_pred))

    # Generating Confusion Matrix
    print('Confusion Matrix:')
    cf_matrix = confusion_matrix(y_val, y_pred)
    print(cf_matrix)

    # Visualizing Confusion Matrix with Labels
    plt.title('Confusion matrix of ' + title + ' classifier')
    group_names = ['True Neg', 'False Pos', 'False Neg', 'True Pos']
    group_counts = ["{0:0.0f}".format(value) for value in cf_matrix.flatten()]
    group_percentages = ["{0:.2%}".format(
        value) for value in cf_matrix.flatten()/np.sum(cf_matrix)]
    labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in zip(
        group_names, group_counts, group_percentages)]
    labels = np.asarray(labels).reshape(2, 2) 
    ax = sns.heatmap(cf_matrix, annot=labels, fmt='', cmap='Blues')
    ax.set(ylabel="Actual Label", xlabel="Predicted Label") 
    plt.show()

    # Save the model
    path = "data/" + str(name) + ".model"
    with open(path, 'wb') as handle:
        pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)
