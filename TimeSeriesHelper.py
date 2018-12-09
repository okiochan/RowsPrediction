import numpy as np

def GetRunning(series, start, end):
    res = []
    for i in range(start,end+1):
        res.append(series[i])
    return res

def CreateRunningTable(series,k):
    res = []
    l = len(series)
    for i in range(l-k+1):
        tmp = GetRunning(series,i,i+k-1)
        res.append(tmp)
    return np.array(res)

def PrepareLearningTable(y,k):
    x = CreateRunningTable(y,k)
    x = np.array(x)
    y = y[k:]
    x = x[:-1,:]
    return x,y

def TableOneFromAnother(y1,y2,k):
    y1 = CreateRunningTable(y1,k)
    return y1, np.array(y2[k-1:])