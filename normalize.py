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

def Lerp(t,y1,y2):
    return (1-t)*y1 + t*y2

def Interpolate(x1,y1,x2,y2,x):
    t = (x - x1) / (x2 - x1)
    y = Lerp(t,y1,y2)
    return y

def FillMissingValues(x1,y1,x2,y2,tol=1e-10):
    d1 = x1.max()-x1.min()
    d2 = x2.max()-x2.min()
    d = max(d1,d2)
    newx = []
    newy1 = []
    newy2 = []
    i = 0
    j = 0
    while i < x1.size and j < x2.size:
        if abs(x1[i]-x2[j]) <= d*tol:
            newx.append((x1[i]+x2[j])/2)
            newy1.append(y1[i])
            newy2.append(y2[j])
            i+=1
            j+=1
        elif x1[i] < x2[j]:
            if j-1>=0:
                y = Interpolate(x2[j-1],y2[j-1],x2[j],y2[j],x1[i])
                newx.append(x1[i])
                newy1.append(y1[i])
                newy2.append(y)
            i+=1
        elif x2[j] < x1[i]:
            if i-1 >=0:
                y = Interpolate(x1[i-1],y1[i-1],x1[i],y1[i],x2[j])
                newx.append(x2[j])
                newy1.append(y)
                newy2.append(y2[j])
            j+=1
    return np.array(newx,dtype=float),np.array(newy1,dtype=float),np.array(newy2,dtype=float)
