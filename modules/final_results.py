from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score, recall_score, roc_curve, auc, roc_auc_score, precision_score
from sklearn.model_selection import train_test_split
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle, time

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

def get_res(best_k, model_name, scores_n_timings_only = False, title=''):
    # Load the data
    path = "data/train_test.data"
    with open(path, 'rb') as handle:
        data = pickle.load(handle)
    
    X_train = data['X_train']
    X_val = data['X_val']
    y_train = data['y_train']
    y_val = data['y_val']

    # Load the model
    path = 'data/' + str(model_name) + '.model'
    with open(path, 'rb') as handle:
        model = pickle.load(handle)

    # Get Accuracy
    start_time = time.time()
    y_pred = model.predict(X_val)
    end_time = time.time()

    print("Number of mislabeled points out of a total %d points : %d" % (X_val.shape[0], (y_val != y_pred).sum()))
    ac_score = accuracy_score(y_val, y_pred)
    print("Accuracy is ", ac_score)
    f1_sc = f1_score(y_val, y_pred)
    print("F-Score is ", f1_sc)
    rec_score = recall_score(y_val, y_pred)
    print("Recall Score is ", rec_score)
    pre_sc = precision_score(y_val, y_pred)
    print("Precision Score is ", pre_sc)
    if (scores_n_timings_only):
        inference_time = (end_time - start_time) / len(y_pred)
        return ac_score, f1_sc, rec_score, pre_sc, inference_time
    
    # Conf matrix
    show_conf_matrix(y_val, y_pred, title)


    # Get predicted and true probabilities
    y_score = model.predict_proba(X_val)
    # print(y_score)
    # print(type(y_score))
    y_test = []
    for y in y_val:
        if int(y) == 1:
            y_test.append([0.0, 1.0])
        else:
            y_test.append([1.0, 0.0])
    y_test = np.array(y_test)


    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(2):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Plot ROC for positive
    class_names = ["Negative", "Positive"]
    for i in range(2): 
        plt.figure()
        lw = 2
        plt.plot(fpr[i], tpr[i], color='darkorange',
                lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[i])
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False ' + class_names[i] + ' Rate')
        plt.ylabel('True ' + class_names[i] + ' Rate')
        plt.title('Receiver operating characteristic for ' + title)
        plt.legend(loc="lower right")
        plt.show()