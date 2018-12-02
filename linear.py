import numpy as np
import matplotlib.pyplot as plt
import dataRidge
import normalize

def RidgeRegression(X,y,C):
    l = X.shape[0]
    n = X.shape[1]

    # bias trick - concatenate ones in front of matrix
    ones = np.atleast_2d(np.ones(l)).T
    X = np.concatenate((ones,X),axis=1)

    # learn linear MNK
    res = np.linalg.inv(X.T.dot(X) + np.eye(n+1) * C).dot(X.T.dot(y))
    return res[1:(n+1)], res[0]

def SSE(X,y,a,b):
    l = X.shape[0]
    loss = 0
    for i in range(l):
        loss += (a.dot(X[i,...])+b-y[i])**2
    return loss * 0.5 / l

def newSample(X,Y,k):
    n = Y.shape[0]
    Y2 = np.zeros((n-k+1, k))

    for i in range(n-k+1):
        for j in range(k):
            Y2[i,j] = Y[i+j]

    X = X[kgram:]
    Y = Y[kgram:]
    tmp = Y2.shape[0]
    Y2 = Y2[:tmp-1,:]
    return X,Y,Y2


    #propagate Y2 by Y1; SAME DIMENSIONS!
def Propagate(X1, Y1, X2, Y2, kgram):
    X1, Y1, Y1new = newSample(X1,Y1,kgram)
    X2, Y2, Y2new = newSample(X2,Y2,kgram)
    w, w0 = RidgeRegression(Y1new,Y2,C=1e-9)
    print(w,w0)

    n = X1.shape[0]
    Yhat = np.zeros(n)
    for i in range(n):
        Yhat[i] = w.dot(Y1new[i,...])+w0
    print("SSE: ", SSE(Y1new,Y2,w,w0))
    return X1, Y1, X2, Y2, Yhat




#-----------------------------------------------------------------
                    # propagate RowC by RowA

X1, Y1 = dataRidge.DataBuilder().Build("RowB")
X2, Y2 = dataRidge.DataBuilder().Build("RowC")
X1 = X1.reshape(-1,1)
X2 = X2.reshape(-1,1)
X1, Y1 = normalize.Normalize(X1, Y1)
X2, Y2 = normalize.Normalize(X2, Y2)

plt.plot(X1.ravel(),Y1, c = "green")
plt.plot(X2.ravel(),Y2, c = "blue")
plt.show()

X1 = np.concatenate((X1[0:11],X1[20:]))
Y1 = np.concatenate((Y1[0:11],Y1[20:]))
# print(X1.shape[0], X2.shape[0])
# quit()


kgram = 3
X1, Y1, X2, Y2, Yhat = Propagate(X1, Y1, X2, Y2, kgram)

plt.plot(X2,Y2, c = "blue")
plt.plot(X2, Yhat, c='orange')
plt.show()


