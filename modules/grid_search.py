import copy, os, json
def test(a, b):
    print(a, b)

def add_to_grid(params, grid, grid_val = {}):
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

def grid_search(params, name = None):
    GRID_PATH = "grid_search_data/" + name + "_grid.json"
    GRID_EXISTS = False

    # check if there is any saved grid
    if name != None:
        if os.path.exists(GRID_PATH):
            GRID_EXISTS = True
            with open(GRID_PATH) as f:
                grid = json.load(f)
        

    if not GRID_EXISTS:
        grid = list()
        add_to_grid(params, grid)
        # print(grid)
        try:
            os.mkdir("grid_search_data")
        except FileExistsError:
            pass
        with open(GRID_PATH, 'w') as f:
            json.dump(grid, f)
    

if __name__ == "__main__":
    # param = {"a": 1, "b": 2}
    # test(**param)

    params = {
        "a": [1, 2, 0],
        "b": [3, 4],
        "c": [5, 6],
        "d": [10, 20, 30, 40, 50]
    }
    grid_search(params, "xgb2")