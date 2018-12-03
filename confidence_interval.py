import numpy as np

def Solve(X,Y):
    H = X.T.dot(X) + 1e-8
    b = X.T.dot(Y)
    return np.linalg.solve(H,b)

def GetDistribution(X,Y,samples=1000):
    l = X.shape[0]
    allW = []
    for s in range(samples):
        choice = np.random.choice(np.arange(l),l)
        Xsel = X[choice,:]
        Ysel = Y[choice]
        allW.append(Solve(Xsel,Ysel))
    return np.array(allW,dtype=float)

def GetErrorDistribution(X,Y):
    w = Solve(X,Y)
    l = X.shape[0]
    errs = []
    for i in range(l):
        errs.append(X.dot(w)-Y)
    return np.array(errs,dtype=float).ravel()


# def GetErrorQuantile(X,Y,quantile=0.5):
    # w = Solve(X,Y)
    # l = X.shape[0]
    # errs = []
    # for i in range(l):
        # errs.append(X.dot(w)-Y)
    # return np.array(errs,dtype=float)