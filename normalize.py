import numpy as np

def NormalizeVec(x):
    return 2*(x-x.min())/(x.max()-x.min())-1

def Normalize(X,Y):
    X = X.copy()
    Y = Y.copy()
    n = X.shape[1]
    for j in range(n):
        X[:,j]=NormalizeVec(X[:,j])
    Y = NormalizeVec(Y)
    return X,Y