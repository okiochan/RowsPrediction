import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

def PlotTimeSeries(y,offset,**kwargs):
    plt.plot(np.arange(y.size)+offset,y,**kwargs)

def PlotPrediction(x,w,offset,**kwargs):
    yHat = np.dot(x,w)
    PlotTimeSeries(yHat,offset,**kwargs)

def DensityPlot(data,**kwargs):
    from scipy.stats import gaussian_kde
    density = gaussian_kde(data)
    d = np.max(data)-np.min(data)

    xs = np.linspace(np.min(data)-d*0.2,np.max(data)+d*0.2,200)
    density._compute_covariance()
    y = density(xs)
    y /= np.abs(y).max()
    plt.plot(xs,y,**kwargs)

def AddLegends(colors,names):
    patches=[]
    for it in zip(names,colors):
        patches.append(mpatches.Patch(color=it[1], label=it[0]))
    plt.legend(handles=patches)