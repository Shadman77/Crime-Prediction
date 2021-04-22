from modules import grid_search, train_best
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier
import json
if __name__ == "__main__":
    # Value of k used in grid search
    k = "5"

    # XGB
    best_results, best_params = grid_search.get_best_params("XGB")    
    param = json.loads(best_params[k])
    train_best.train_save_without_k_fold(XGBClassifier, param, "xgb_best")
    print("XGB Done")

    # RF
    best_results, best_params = grid_search.get_best_params("RF")
    param = json.loads(best_params[k])
    param['n_jobs'] = -1
    train_best.train_save_without_k_fold(RandomForestClassifier, param, "rf_best")
    print("RF Done")

    # AB
    best_results, best_params = grid_search.get_best_params("AB")
    param = json.loads(best_params[k])
    train_best.train_save_without_k_fold(AdaBoostClassifier, param, "ab_best")
    print("AB Done")

    # GNB
    best_results, best_params = grid_search.get_best_params("GNB")
    param = json.loads(best_params[k])
    train_best.train_save_without_k_fold(GaussianNB, param, "gnb_best")
    print("GNB Done")