import numpy as np 
import pandas as pd

X = pd.read_csv("C1_Data_Test.csv", header=None)
X = X.values

Y = X[:, 7]
X = X[:,0:7]
W = np.zeros((X.shape[1]))

W[0] = 1.78791026e-01
W[1] = 4.31649340e+00
W[2] = 3.64352954e+00
W[3] = 4.16847883e+03
W[4] = -2.75098857e+03
W[5] = 2.88177166e+02
W[6] = 3.82136128e+02

y = X.dot(W)
print("Cost Matrix : \n", W)
print("\n\n\nOriginal Data Y : \n", Y)
print("\n\n\nFit Data : \n", y)

error = 0
for i in range(0, Y.shape[0]):
	error = error + (Y[i] - y[i])**2

error = error / Y.shape[0];
print("\n\n\n\nMean Squared Error : ", error)
