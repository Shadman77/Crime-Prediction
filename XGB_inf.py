import pickle, numpy, utils

if __name__ == "__main__":

    # Load the saved model
    with open('data/xgb.model', 'rb') as handle:
        xgb = pickle.load(handle)

    # Load example input
    with open('data/xgb_example.pd', 'rb') as handle:
        X = pickle.load(handle)

    # Get result using the example input
    result = xgb.predict(X)
    print("The crime is:", utils.interpret_res(result[0]))

    #Set value 
    print()
    X = X.reset_index(drop=True)
    # print(X)
    for column in X.iloc[0].keys():
        print(column, X.at[0, column], type(X.at[0, column]))
        value = input("Enter the value of " + column + ": ")

        if isinstance(X.at[0, column], numpy.int64):
            X.at[0, column] = numpy.int64(int(value))
        elif isinstance(X.at[0, column], numpy.float64):
            X.at[0, column] = numpy.float64(float(value))
        elif isinstance(X.at[0, column], numpy.bool):
            if value.lower() in ['true', '1', 'yes']:
                X.at[0, column] = numpy.bool(True)
            else:
                X.at[0, column] = numpy.bool(False)

        print(column, X.at[0, column], type(X.at[0, column]))
        print()


    # Get result using the custom input
    result = xgb.predict(X)
    print("The crime is:", utils.interpret_res(result[0]))