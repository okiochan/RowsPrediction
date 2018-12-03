import matplotlib.pyplot as plt
import numpy as np

def Test_GetData():
    import dataRidge
    X1, Y1 = dataRidge.DataBuilder().Build("RowB")
    plt.plot(X1,Y1)
    plt.show()

def Test_FillMissingValues():
    import normalize
    x1 = np.linspace(-6,6,10)
    y1 = np.sin(x1)
    x2 = np.linspace(-5,5,50)
    y2 = np.cos(x2)

    x3,y1new,y2new = normalize.FillMissingValues(x1,y1,x2,y2)

    plt.plot(x1,y1)
    plt.plot(x2,y2)
    plt.scatter(x3,y1new,c="r")
    plt.scatter(x3,y2new,c="g")
    plt.show()

    quit()

def Test_ConfidenceInterval():
    import confidence_interval
    import dataRidge
    X1, Y1 = dataRidge.DataBuilder().Build("RowB")
    X1 = X1.reshape(-1,1)
    X1 = dataRidge.AddOnes(X1)
    allW = confidence_interval.GetDistribution(X1,Y1)
    print(allW)
    for i in range(allW.shape[1]):
        plt.hist(allW[:,i],bins=30)
        plt.show()
        
    errs = confidence_interval.GetErrorDistribution(X1,Y1)
    print(np.percentile(errs,0))
    print(np.percentile(errs,50))
    print(np.percentile(errs,95))
    print(np.percentile(errs,100))
    plt.hist(errs,bins=10)
    plt.show()


if __name__ == "__main__":
    # Test_FillMissingValues()
    # Test_GetData()
    Test_ConfidenceInterval()
    pass