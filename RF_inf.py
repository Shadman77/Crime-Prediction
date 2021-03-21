import pickle, numpy, utils
from modules import options

if __name__ == "__main__":

    # Load the saved model
    with open('data/rf.model', 'rb') as handle:
        rf = pickle.load(handle)

    # Load example input
    with open('data/xgb_example.pd', 'rb') as handle:
        X = pickle.load(handle)

    # Get result using the example input
    result = rf.predict(X)
    print("The crime is:", utils.interpret_res(result[0]))

    #Set value 
    print()
    X = X.reset_index(drop=True)
    # print(X)
    for column in X.iloc[0].keys():
        # break
        if "Weapon_" in column:
            continue
        if "Loc_Dec_" in column:
            continue
        if "Perp_Race_" in column:
            continue
        if "Perp_Ethnicity_" in column:
            continue
        if "Perpetrator Sex" == column:
            continue

        print(column, X.at[0, column], type(X.at[0, column]))
        value = input("Enter the value of " + column + ": ")

        if isinstance(X.at[0, column], numpy.int64):
            X.at[0, column] = numpy.int64(int(value))
        elif isinstance(X.at[0, column], numpy.float64):
            X.at[0, column] = numpy.float64(float(value))
        elif isinstance(X.at[0, column], numpy.bool_):
            if value.lower() in ['true', '1', 'yes']:
                X.at[0, column] = True
            else:
                X.at[0, column] = False

        print(column, X.at[0, column], type(X.at[0, column]))
        print()
    
    # Weapon
    X = options.select_weapon(X)

    # Location
    X = options.select_loc_desc(X)

    # Sex
    X = options.select_sex(X)

    # Race
    X = options.select_race(X)

    # Ethnicity
    X = options.select_ethnicity(X)

    # Get result using the custom input
    result = rf.predict(X)
    print("The crime is:", utils.interpret_res(result[0]))