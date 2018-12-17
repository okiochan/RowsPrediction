import numpy as np
import LeastSquares

#raspred coeffs
def GetDistribution(X,Y,samples=1000):
    l = X.shape[0]
    allW = []
    for s in range(samples):
        choice = np.random.choice(np.arange(l),l)
        Xsel = X[choice,:]
        Ysel = Y[choice]
        allW.append(LeastSquares.Solve(Xsel,Ysel))
    return np.array(allW,dtype=float)

#raspred errors
def GetErrorDistribution(X,Y):
    w = LeastSquares.Solve(X,Y)
    l = X.shape[0]
    errs = []
    for i in range(l):
        errs.append(X.dot(w)-Y)
    return np.array(errs,dtype=float).ravel()

