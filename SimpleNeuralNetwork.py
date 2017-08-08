import pandas as pd
import numpy as np

# Read CSF file into pandas dataframe
inputDataColumnsToBeRead = ['TV', 'Radio']
advertisingData = pd.read_csv('C:\Developer\MachineLearning\MachineLearningInPython\Supportingfiles\Advertising.csv', usecols=inputDataColumnsToBeRead)

outputColumn = ['Sales']
salesData =  pd.read_csv('C:\Developer\MachineLearning\MachineLearningInPython\Supportingfiles\Advertising.csv', usecols=outputColumn)

# Convert pandas dataframe into input matrix and output vector
x = np.array(advertisingData)
y = np.array(salesData)

# number of Rows and columns in Input dataframe
numberOfRowsInInputData = x.shape[0]
numberOfColumnsInInputData = x.shape[1]

# number of nodes in hidden layer
numberOfNodesInHiddenLayer = 3

# initialize weight for first layer
theta1 = np.empty([numberOfColumnsInInputData, numberOfNodesInHiddenLayer])
theta1.fill(0.1)

# initialize weight for second layer
theta2 = np.empty([numberOfNodesInHiddenLayer, 1])
theta2.fill(0.01)

# Activation/Sigmoid Function Definition
def sigmoidFunction(x):
    return 1/(1 + np.exp(x))

# Derivative for sigmoid Function
def derivativeSigmoidFunction(x):
    return np.exp(-x)/np.square((1 + np.exp(-x)))

for index in range(1000):
    # compute hidden layer
    z1 = np.dot(x, theta1)

    # apply activation(sigmoid) on hidden layer
    a1 = sigmoidFunction(-z1)

    # compute final layer
    z2 = np.dot(a1, theta2)

    # compute predicted output(apply sigmoid)
    predictedY = sigmoidFunction(-z2)

    # Back propogation error at final layer
    # Calculate Error
    error = y - predictedY

    # determine slopes at output and hidden later
    slope_output_layer = derivativeSigmoidFunction(z2)
    slope_hidden_layer = derivativeSigmoidFunction(z1)

    # Change factor at output layer
    delta3 = -(error)*slope_output_layer
    #delta3 = -np.dot(error, slope_output_layer)

    # compute error at output layer
    errorTheta2 = np.dot(np.transpose(a1), delta3)
    theta2 = theta2 - errorTheta2

    # compute error at hidden layer
    delta2 = delta3*np.transpose(theta2)*slope_hidden_layer

    # compute error at hidden layer
    errorTheta1 = np.dot(np.transpose(x), delta2)
    theta1 = theta1 - errorTheta1
