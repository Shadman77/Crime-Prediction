import copy
import os
import json
import sklearn


def add_to_grid(params, grid, grid_val={}):
    # get a single param from the dictionary params
    param_name = list(params.keys())[0]
    param = params[param_name]

    # create another set of params without the current param
    new_params = copy.deepcopy(params)
    del new_params[param_name]

    # for each value in the param
    for value in param:
        # append the value of the param in the single grid value, here grid_val is a dictionary
        grid_val[param_name] = value
        # if this is the last param
        if len(params.keys()) == 1:
            # we print the single grid value
            # print(grid_val)
            grid.append(copy.deepcopy(grid_val))
        else:
            # we recursively move to another param
            add_to_grid(new_params, grid, grid_val)


def train_for_grid(Model, param, ks, X_train, y_train):
    results = {}

    # for each value of k perform k-fold CV
    for k in ks:
        print("K = ", k)
        model = Model(**param)
        kfold = sklearn.model_selection.KFold(n_splits=k)
        single_results = sklearn.model_selection.cross_val_score(model,
                                                                 X_train,
                                                                 y_train,
                                                                 cv=kfold,
                                                                 verbose=2,
                                                                 n_jobs=2)

        results[str(k)] = [single_results.mean() *
                           100.0, single_results.std()*100.0]

    print(results)
    return results


def grid_search(params, Model, X_train, y_train, name=None):
    GRID_PATH = "grid_search_data/" + name + "_grid.json"
    RESULTS_PATH = "grid_search_data/" + name + "_results.json"
    GRID_EXISTS = False
    RESULTS_EXISTS = False
    KS = [10, 15, 20]

    # Create folder if already does not exist
    try:
        os.mkdir("grid_search_data")
    except FileExistsError:
        pass

    # check if there is any saved grid and results
    if name != None:
        if os.path.exists(GRID_PATH):
            GRID_EXISTS = True
            with open(GRID_PATH) as f:
                grid = json.load(f)

        if os.path.exists(RESULTS_PATH):
            RESULTS_EXISTS = True
            with open(RESULTS_PATH) as f:
                results = json.load(f)

    if not GRID_EXISTS:
        grid = list()
        add_to_grid(params, grid)
        # print(grid)
        with open(GRID_PATH, 'w') as f:
            json.dump(grid, f)

    if not RESULTS_EXISTS:
        results = {}

    for single_param in grid:
        result_key = json.dumps(single_param)
        print(result_key)
        if not results.__contains__(result_key):
            print("Training...")
            results[result_key] = train_for_grid(
                Model, single_param, KS, X_train, y_train)
            with open(RESULTS_PATH, 'w') as f:
                json.dump(results, f)


def get_best_params(name):
    # print("Getting best params")
    RESULTS_PATH = "grid_search_data/" + name + "_results.json"
    KS = [10, 15, 20]
    k_max = {}
    k_max_param = {}

    for k in KS:
        k_max[str(k)] = 0
        k_max_param[str(k)] = "" 

    try:
        with open(RESULTS_PATH) as f:
            results = json.load(f)
    except:
        print("Could not load results")

    for result in results:
        # print(results[result])
        single_result = results[result]
        for k in KS:
            if single_result[str(k)][0] > k_max[str(k)]:
                k_max[str(k)] = single_result[str(k)][0]
                k_max_param[str(k)] = result
    
    # print(k_max)
    # print(k_max_param)

    return k_max, k_max_param


if __name__ == "__main__":
    params = {
        "eta": [0.3, 0.4],
        "max_depth": [6, 10],
    }
    grid_search(params, "", "xgb2")
