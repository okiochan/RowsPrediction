import numpy as np
import matplotlib.pyplot as plt
import dataRidge
import normalize
import TimeSeriesHelper
import LeastSquares
import ConfidenceInterval
import PlottingHelper

def SSE(X,y,a,b):
    l = X.shape[0]
    loss = 0
    for i in range(l):
        loss += (a.dot(X[i,...])+b-y[i])**2
    return loss * 0.5 / l

#build that matrix
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
    #make the matrix below
    X1, Y1, Y1new = newSample(X1,Y1,kgram)
    X2, Y2, Y2new = newSample(X2,Y2,kgram)
    #get w,w0 with linear regression
    w, w0 = RidgeRegression(Y1new,Y2,C=1e-9)
    print(w,w0)

    n = X1.shape[0]
    Yhat = np.zeros(n)
    for i in range(n):
        Yhat[i] = w.dot(Y1new[i,...])+w0
    print("SSE: ", SSE(Y1new,Y2,w,w0))
    return X1, Y1, X2, Y2, Yhat


def MakePredictions(last,need,w,err=0):
    res = []
    now = last
    n = w.size-1
    for i in range(need):
        res.append(np.dot(now,w))
        nxt = []
        for j in range(n-1):
            nxt.append(now[j+1])
        nxt.append(res[-1])
        nxt.append(1)
        nxt = np.array(nxt)
        now = nxt
    for i in range(len(res)):
        res[i] += (i+1)*err
    return res


#ideal: k=2, Sin, Cos
k = 4
needPreds = 50
# X1, Y1 = dataRidge.DataBuilder().Build("GetQuadraticTrendDown")
# X2, Y2 = dataRidge.DataBuilder().Build("GetQuadraticTrendUp")

# X1, Y1 = dataRidge.DataBuilder().Build("helloSin")
# X2, Y2 = dataRidge.DataBuilder().Build("helloCos")

# X1, Y1 = dataRidge.DataBuilder().Build("SalesA")
# X2, Y2 = dataRidge.DataBuilder().Build("SalesB")

# X1, Y1 = dataRidge.DataBuilder().Build("RowApred")
# X1, Y1 = dataRidge.DataBuilder().Build("RowB")
# X2, Y2 = dataRidge.DataBuilder().Build("RowC")

# X1, Y1 = dataRidge.DataBuilder().Build("SalesWaltmartApred")
X1, Y1 = dataRidge.DataBuilder().Build("SalesWaltmartA")
X2, Y2 = dataRidge.DataBuilder().Build("SalesWaltmartB")

# plt.plot(X1,Y1, c="blue")
# plt.plot(X2,Y2, c="orange")
# plt.show()
# quit()


#interpolation
X,Y1,Y2 = normalize.FillMissingValues(X1,Y1,X2,Y2)

#[-1,1]
nrmX = normalize.Normalizer(X)
nrmY1 = normalize.Normalizer(Y1)
nrmY2 = normalize.Normalizer(Y2)
X = nrmX(X)
Y1 = nrmY1(Y1)
Y2 = nrmY2(Y2)

#Linear prediction
#create table for learning
x,y = TimeSeriesHelper.PrepareLearningTable(Y1,k)
x = dataRidge.AddOnes(x)
w = LeastSquares.Solve(x,y)
errs = ConfidenceInterval.GetErrorDistribution(x,y)

# yPred = []
# for i in range(y.size):
    # yPred.append(np.dot(w,x[i,:]))
# yPredUp = yPred[:-1] + MakePredictions(x[-1,:],needPreds,w,np.max(errs))
# yPredLow = yPred[:-1] + MakePredictions(x[-1,:],needPreds,w,np.min(errs))
# yPred05 = yPred[:-1] + MakePredictions(x[-1,:],needPreds,w,np.percentile(errs,10))
# yPred95 = yPred[:-1] + MakePredictions(x[-1,:],needPreds,w,np.percentile(errs,90))
# yPredMean = yPred[:-1] + MakePredictions(x[-1,:],needPreds,w)
# _, yPredMean = dataRidge.DataBuilder().Build("SalesApred")
_, yPredMean = dataRidge.DataBuilder().Build("SalesWaltmartApred")
yPredMean = nrmY1(yPredMean)
# print(yPredMean)
# quit()

A,B = TimeSeriesHelper.TableOneFromAnother(Y1,Y2,k)
A = dataRidge.AddOnes(A)
w2 = LeastSquares.Solve(A,B)
A2 = TimeSeriesHelper.CreateRunningTable(yPredMean,k)
A2 = dataRidge.AddOnes(A2)
B2 = np.dot(A2,w2)
errs2 = ConfidenceInterval.GetErrorDistribution(A,B)

plt.subplot(2, 1, 1)
wDist = ConfidenceInterval.GetDistribution(x,y)
PlottingHelper.AddLegends(['royalblue','orange'],[
    'First Series Error Density', 'Second Series Error Density'])
PlottingHelper.DensityPlot(errs,c='royalblue')
PlottingHelper.DensityPlot(errs2,c='orange')
# plt.show()
plt.subplot(2, 1, 2)
for i in range(wDist.shape[1]):
    PlottingHelper.DensityPlot(wDist[:,i])
plt.show()

plt.subplot(2, 1, 1)
PlottingHelper.AddLegends(['r','g','orange','violet','royalblue'],[
    'Cummulative Max PI',
    'Cummulative 80% PI',
    '80% Percentile PI',
    'Mean',
    'Points'])
plt.scatter(np.arange(y.size),y,linewidth=2.0,c='royalblue')
# plt.plot(np.arange(len(yPredMean)),yPredUp,c='r')
# plt.plot(np.arange(len(yPredMean)),yPredLow,c='r')
# plt.plot(np.arange(len(yPredMean)),yPred05,c='g')
# plt.plot(np.arange(len(yPredMean)),yPred95,c='g')
plt.plot(np.arange(len(yPredMean)),yPredMean+np.percentile(errs,10),c='orange')
plt.plot(np.arange(len(yPredMean)),yPredMean+np.percentile(errs,90),c='orange')
plt.plot(np.arange(len(yPredMean)),yPredMean,c='violet')
plt.ylim(ymin=-1.2,ymax=1.2)

plt.subplot(2, 1, 2)
PlottingHelper.AddLegends(['royalblue','orange','green'],[
    'Points',
    'Mean',
    '80% Percentile PI'])
plt.scatter(np.arange(Y2.size),Y2,c='royalblue')
PlottingHelper.PlotTimeSeries(B2,2*k-1,c='orange')
PlottingHelper.PlotTimeSeries(B2 + np.percentile(errs2,10),2*k-1,c='green')
PlottingHelper.PlotTimeSeries(B2 + np.percentile(errs2,90),2*k-1,c='green')
plt.show()