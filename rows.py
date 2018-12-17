import numpy as np
import matplotlib.pyplot as plt
import dataRidge
import normalize


#ideal: k=2, Sin, Cos
k = 3
needPreds = 50

#X1, Y1 = dataRidge.DataBuilder().Build("SalesA")
#X2, Y2 = dataRidge.DataBuilder().Build("SalesB")

# X1, Y1 = dataRidge.DataBuilder().Build("RowA")
# X2, Y2 = dataRidge.DataBuilder().Build("RowC")

X1, Y1 = dataRidge.DataBuilder().Build("RowB")
X2, Y2 = dataRidge.DataBuilder().Build("RowC")

#interpolation
X,Y1,Y2 = normalize.FillMissingValues(X1,Y1,X2,Y2)

#[-1,1]
X = normalize.NormalizeVec(X)
Y1 = normalize.NormalizeVec(Y1)
Y2 = normalize.NormalizeVec(Y2)


plt.scatter(X,Y1)
plt.show()