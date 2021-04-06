from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import train_test_split
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle

def show_conf_matrix(y_val, y_pred, title):
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

def get_res(best_k, model_name, title):
    # Load the dataset
    df = pd.read_csv('data/smote.csv')

    # Seperate X and y
    X = df.drop(columns=["Primary Type"])
    y = df["Primary Type"]
    
    # Train test split using the value of k as ratio ref
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=1/float(best_k))

    # Load the model
    path = 'data/' + str(model_name) + '.model'
    with open('data/rf.model', 'rb') as handle:
        model = pickle.load(handle)

    # Get Accuracy
    y_pred = model.predict(X_val)
    print("Number of mislabeled points out of a total %d points : %d" % (X_val.shape[0], (y_val != y_pred).sum()))
    print("Accuracy is ", accuracy_score(y_val, y_pred))
    show_conf_matrix(y_val, y_pred, title)