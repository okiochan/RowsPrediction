import numpy as np

def Solve(X,Y,ridge=1e-4):
    H = X.T.dot(X) + ridge
    b = X.T.dot(Y)
    return np.linalg.solve(H,b)
