import numpy as np
import pandas as pd

# X = np.array([ [0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1] ])
X = np.random.random((4, 3))
y = np.array( [[1, 0, 1, 0]] ).T

def sigmoid(X):
    return ( 1 / (1 + np.exp(-X) ) )

def tanh(X):
    return 2 * sigmoid(2 * X) - 1

def dsigmoid(X):
    return sigmoid(X) * (1 - sigmoid(X))

def dtanh(X):
    return 1 - np.power(tanh(X), 2)

weight0 = 2 * np.random.random((3, 4)) - 1
weight1 = 2 * np.random.random((4, 4)) - 1
weight2 = 2 * np.random.random((4, 1)) - 1

learning_rate = 0.1

for iteration in range(20000):

    # forward:

    layor1 = sigmoid(np.dot(X, weight0)) 
    layor2 = sigmoid(np.dot(layor1, weight1)) 
    layor3 = sigmoid(np.dot(layor2, weight2)) 

    # backward:

    dlayor3 = (y - layor3) * dsigmoid(np.dot(layor2, weight2)) 
    dlayor2 = np.dot(dlayor3, weight2.T) * dsigmoid(np.dot(layor1, weight1)) 
    dweight2 = np.dot(layor2.T, dlayor3)
    dlayor1 = np.dot(dlayor2, weight1.T) * dsigmoid(np.dot(X, weight0)) 
    dweight1 = np.dot(layor1.T, dlayor2)
    dX = np.dot(dlayor1, weight0.T)
    dweight0 = np.dot(X.T, dlayor1)

    # update:

    weight0 += dweight0 * learning_rate
    weight1 += dweight1 * learning_rate
    weight2 += dweight2 * learning_rate

print(layor3)
