import matplotlib.pyplot as plt
import numpy as np
import scipy.misc
import math
from sklearn import datasets
import pandas as pd 

def AddOnes(X):
    l = X.shape[0]
    ones = np.ones((l,1))
    X = np.concatenate((X,ones),axis=1)
    return X

class FakeData:
    def GenerateSample(self):
        l = 50
        real = 2
        X = np.random.randn(l,real)
        a = np.random.randn(real)
        y = np.zeros(l)
        for i in range(l):
            y[i] = a.dot(X[i,:])

        # fake parameters
        F = 2 * X
        Xp = np.concatenate((X,F),axis=1)
        return Xp, y

class SkikitData:
    def GenerateSample(self):
        print("hello")
        X = 1. / (np.arange(1, 11) + np.arange(0, 10)[:, np.newaxis])
        Y = np.ones(10)
        return X,Y

class HelloSin:
    def GenerateSample(self):
        X = np.linspace(-10,10,100)
        Y = np.sin(X)
        return X.reshape(-1,1),Y


class HelloCos:
    def GenerateSample(self):
        X = np.linspace(-10,10,100)
        Y = np.cos(X)
        return X.reshape(-1,1),Y


class RowA:
    def GenerateSample(self):
        arr =  [1950, 646.4, 1955, 741.6, 1960, 556.6, 1965, 623.7, 1970, 558.1, 1975, 622.9,
        1980, 610.1, 1985, 574.4, 1990, 548.4, 1995, 561.7, 2000, 546.1, 2001, 643.6,
        2002, 611.2, 2003, 536.9, 2004, 562.4, 2005, 568.8, 2006, 506.8, 2007, 571.8,
        2008, 615.0, 2009, 698.6, 2010, 678.2, 2011, 606.6, 2012, 563.9, 2013, 527.5,
        2014, 514.7, 2015, 510.9, 2016, 500.7]

        X = arr[0::2]
        Y = arr[1::2]
        X = np.array(X,dtype = float)
        Y = np.array(Y,dtype = float)
        return X,Y
        

class RowApred:
    def GenerateSample(self):
        arr =  [1950, 646.4, 1955, 741.6, 1960, 556.6, 1965, 623.7, 1970, 558.1, 1975, 622.9,
        1980, 610.1, 1985, 574.4, 1990, 548.4, 1995, 561.7, 2000, 546.1, 2001, 643.6,
        2002, 611.2, 2003, 536.9, 2004, 562.4, 2005, 568.8, 2006, 506.8, 2007, 571.8,
        2008, 615.0, 2009, 698.6, 2010, 678.2, 2011, 606.6, 2012, 563.9, 2013, 527.5,
        2014, 514.7, 2015, 510.9, 2016, 500.7]

        X = arr[0::2]
        Y = arr[1::2]
        X = np.array(X,dtype = float)
        Y = np.array(Y,dtype = float)
        
        Y1 = [469.8064731635153, 427.48114275895057, 602.4281668137759, 698.0441552106291, 530.5721172675491, 588.6991096416489, 508.306023612991, 585.7824006766081, 566.6801182668656, 539.3085684586316]
        X1 = np.arange(2017,2017+len(Y1))
        
        Y = np.concatenate( (Y,Y1), axis=0)
        X = np.concatenate( (X,X1), axis=0)
        
        return X,Y

class RowB:
    def GenerateSample(self):
        arr =  [1950, 3.9, 1955, 13.4, 1960, 14.1, 1965, 17.1, 1970, 32.7, 1975, 23.2,
       1980, 27.2, 1985, 24.1, 1990, 36.3, 1995, 27.0, 2000, 20.1, 2001, 22.4,
       2002, 19.3, 2003, 14.5, 2004, 22.1, 2005, 21.2, 2006, 24.3, 2007, 22.6,
       2008, 28.3, 2009, 24.4, 2010, 21.2, 2011, 31.9, 2012, 16.9, 2013, 16.1,
       2014, 21.8, 2015, 25.2, 2016, 25.8]

        X = arr[0::2]
        Y = arr[1::2]
        X = np.array(X,dtype = float)
        Y = np.array(Y,dtype = float)
        return X,Y

        
        
class RowBpred:
    def GenerateSample(self):
        arr =  [1950, 3.9, 1955, 13.4, 1960, 14.1, 1965, 17.1, 1970, 32.7, 1975, 23.2,
       1980, 27.2, 1985, 24.1, 1990, 36.3, 1995, 27.0, 2000, 20.1, 2001, 22.4,
       2002, 19.3, 2003, 14.5, 2004, 22.1, 2005, 21.2, 2006, 24.3, 2007, 22.6,
       2008, 28.3, 2009, 24.4, 2010, 21.2, 2011, 31.9, 2012, 16.9, 2013, 16.1,
       2014, 21.8, 2015, 25.2, 2016, 25.8]

        X = arr[0::2]
        Y = arr[1::2]
        X = np.array(X,dtype = float)
        Y = np.array(Y,dtype = float)
        
        Y1 = [17.562784659862515, 14.364704981446263, 4.836011600866913, 17.7605346173048, 16.717813915014265, 5.412073123455048, 29.782986375689507, 23.264819419197735, 1.6928573131561286, 19.41546151638031]
        X1 = np.arange(2017,2017+len(Y1))
        
        Y = np.concatenate( (Y,Y1), axis=0)
        X = np.concatenate( (X,X1), axis=0)
        return X,Y

        
class RowC:
    def GenerateSample(self):
        arr =  [1950, 249.7, 1955, 999.6, 1960, 787.8, 1965, 1061.3, 1970, 1826.4, 1975, 1449.8,
       1980, 1659.8, 1985, 1384.1, 1990, 1988.2, 1995, 1505.2, 2000, 1064.4, 2010, 1403.8,
       2011, 1930.8, 2012, 908.3, 2013, 764.8, 2014, 1102.1, 2015, 1263.1, 2016, 1286.5]

        X = arr[0::2]
        Y = arr[1::2]
        X = np.array(X,dtype = float)
        Y = np.array(Y,dtype = float)
        return X,Y

        
class SalesA:
    def GenerateSample(self):
        df=pd.read_csv('sales.csv', sep=',',header=None)
        Y = list()
        for i in range(len(df.values)): 
            if (df.values[i][0] == 1): 
                Y.append(df.values[i][3])
        # print(Y)
        X = np.arange(0,len(Y))
        return X,Y
        
        
class SalesApred:
    def GenerateSample(self):
        df=pd.read_csv('sales.csv', sep=',',header=None)
        Y = list()
        for i in range(len(df.values)): 
            if (df.values[i][0] == 1): 
                Y.append(df.values[i][3])
        Y1 = [28584.995178353038, 36178.21344493895, 29006.68657510109, 55119.61868237316, 41843.158393040445, 10294.500963756442, 13018.29807087779, 19338.37486820489, 22051.032271752058, 30478.76579317018, 62821.580764391714, 46470.16084704906, 845.3363728466611, 2269.9937107133846, 15319.392391899524, 12550.913037304428, 12797.33914059028, 10592.79146410167, 12744.794863929299, 14670.096734928784, 13917.221874225583, 13717.266590714005, 14585.033421187249, 14696.063980668034, 15851.272140442576, 15662.879066788253, 13755.674145814328, 14518.190105185804, 17532.996235045935, 15517.703366816637]
        
        Y = np.concatenate( (Y,Y1), axis=0)
        
        # print(Y)
        X = np.arange(0,len(Y))
        return X,Y
    
class SalesB:
    def GenerateSample(self):
        df=pd.read_csv('sales.csv', sep=',',header=None) 
        Y = list() 
        for i in range(len(df.values)): 
            if (df.values[i][0] != 1): 
                Y.append(df.values[i][3])
        X = np.arange(0,len(Y))
        return X,Y

def GetQuadraticTrend(a,b,c,x,noise=0.1):
    return a*x**2+b*x+c + np.random.randn(x.size)*noise

class DataBuilder:
    def Build(self, name):
        if name == "fake":
            x, y = FakeData().GenerateSample()
            return x, y
        elif name == "ski":
            print("sdf")
            x, y = SkikitData().GenerateSample()
            return x, y

        elif name == "helloSin":
            x, y = HelloSin().GenerateSample()
            return x, y
        elif name == "helloCos":
            x, y = HelloCos().GenerateSample()
            return x, y

        elif name == "RowA":
            x, y = RowA().GenerateSample()
            return x, y
        elif name == "RowB":
            x, y = RowB().GenerateSample()
            return x, y
        elif name == "RowC":
            x, y = RowC().GenerateSample()
            return x, y
        elif name == "SalesA":
            x, y = SalesA().GenerateSample()
            return x, y
        elif name == "SalesB":
            x, y = SalesB().GenerateSample()
            return x, y
            
            
        elif name == "RowApred":
            x, y = RowApred().GenerateSample()
            return x, y
        elif name == "RowBpred":
            x, y = RowBpred().GenerateSample()
            return x, y
        elif name == "SalesApred":
            x, y = SalesApred().GenerateSample()
            return x, y
            
        elif name == "GetQuadraticTrendUp":
            x = np.linspace(0,2,200)
            return x, GetQuadraticTrend(1,3,6,x)
        elif name == "GetQuadraticTrendDown":
            x = np.linspace(0,2,200)
            return x, GetQuadraticTrend(-5,0,-3,x)
        else:
            assert("Unknown data")