# https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.barh.html#matplotlib.pyplot.barh
# https://matplotlib.org/3.1.1/gallery/ticks_and_spines/ticklabels_rotation.html
# https://datatofish.com/horizontal-bar-chart-matplotlib/

from modules import final_results
from matplotlib import pyplot as plt

accuracy = []
f_score = []
recall_score = []
precision_score = []
inference_times = []
models = ["xgb_best", "ab_best", "rf_best", "gnb_best"]
classifiers = ["XGB", "ADB", "RF", "GNB"]

for model in models:
    ac_score, f1_sc, rec_score, pre_sc, inference_time = final_results.get_res(best_k=20, model_name=model, scores_n_timings_only=True)
    accuracy.append(ac_score)
    f_score.append(f1_sc)
    recall_score.append(rec_score)
    precision_score.append(pre_sc)
    inference_times.append(inference_time)

# print(models, accuracy, f_score, recall_score, precision_score, inference_times)


# Accuracy
plt.style.use('ggplot')
plt.barh(y=classifiers, width=accuracy)
plt.title('Model comparison according to accuracy')
plt.ylabel('Classifiers')
plt.xlabel('Accuracy')
plt.show()

# Precision
plt.style.use('seaborn')
plt.barh(y=classifiers, width=precision_score)
plt.title('Model comparison according to precision')
plt.ylabel('Classifiers')
plt.xlabel('Precision')
plt.show()

# Recall
plt.style.use('Solarize_Light2')
plt.barh(y=classifiers, width=recall_score)
plt.title('Model comparison according to recall score')
plt.ylabel('Classifiers')
plt.xlabel('Recall')
plt.show()

# F-measure
plt.style.use('bmh')
plt.barh(y=classifiers, width=recall_score)
plt.title('Model comparison according to f-measure score')
plt.ylabel('Classifiers')
plt.xlabel('F-measure')
plt.show()

# Inference time
plt.style.use('seaborn-pastel')
plt.plot(classifiers, inference_times, marker="o", linestyle='--')
plt.title('Execution time for each classifier')
plt.ylabel("Time in seconds")
plt.xlabel('Classifiers')
plt.show()