import pandas as pd

def __IURC(x):
    try:
        return int(x)
    except:
        x = list(x)
        x[len(x)-1] = 0
        t = ""
        for c in x:
            t += str(c)
        return int(t)

def __CrimeSolved(x):
    if(x == 'Yes'):
        return 1
    else:
        return 0

def __gender(x):
    if x == 'Male':
        return 1
    elif x == 'Female':
        return 0
        

def format_X(X):
    X['IUCR'] = X['IUCR'].apply(__IURC)
    # print(X['IUCR'].unique())

    X = pd.concat([X, pd.get_dummies(X["Location Description"], prefix='Loc_Dec')], axis=1)
    X = X.drop(columns = ["Location Description"])

    # X['Crime Solved'] = X['Crime Solved'].apply(__CrimeSolved)

    X['Perpetrator Sex'] = X['Perpetrator Sex'].apply(__gender)

    X = pd.concat([X, pd.get_dummies(X["Perpetrator Race"], prefix='Perp_Race')], axis=1)
    X = X.drop(columns = ["Perpetrator Race"])

    X = pd.concat([X, pd.get_dummies(X["Perpetrator Ethnicity"], prefix='Perp_Ethnicity')], axis=1)
    X = X.drop(columns = ["Perpetrator Ethnicity"])

    X = pd.concat([X, pd.get_dummies(X["Weapon"], prefix='Weapon')], axis=1)
    X = X.drop(columns = ["Weapon"])
    
    return X
