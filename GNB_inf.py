import pickle, numpy, utils

if __name__ == "__main__":

    # Load the saved model
    with open('data/gnb.model', 'rb') as handle:
        gnb = pickle.load(handle)

    # Load example input
    with open('data/example_input.pd', 'rb') as handle:
        X = pickle.load(handle)

    # Get result using the example input
    result = gnb.predict([X])
    print("The crime is:", utils.interpret_res(result[0]))

    #Set value 
    print()
    for column in X.keys():
        print(column, X[column], type(X[column]))
        value = input("Enter the value of " + column + ": ")

        if isinstance(X[column], numpy.int64):
            X[column] = numpy.int64(int(value))
        elif isinstance(X[column], numpy.float64):
            X[column] = numpy.float64(float(value))
        elif isinstance(X[column], numpy.int64):
            if value.lower() in ['true', '1', 'yes']:
                X[column] = numpy.bool(True)
            else:
                X[column] = numpy.bool(False)

        print(column, X[column], type(X[column]))
        print()


    # Get result using the custom input
    result = gnb.predict([X])
    print("The crime is:", utils.interpret_res(result[0]))