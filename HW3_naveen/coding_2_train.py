## I have used teh template to modify the solution.
## Using the template provided by Prof Andrew Plummer.

### Copyright 2018 by A.R. Plummer


### import numpy and matplotlib

import numpy as np
import matplotlib.pyplot as plt



### using an np.array to represent out data.  the first column is the
### bias term x0. the second and third columns are the truth values of
### the propositions X and Y.  The fourth column is the truth value of
### the proposition X and Y, which we take to be the class value C.

andFunction = np.array([[1,1,1,1],
                       [1,0,1,0],
                       [1,1,0,0],
                       [1,0,0,0]])



### the lines below are used to plot the data. 

for i in range(len(andFunction)):
    plt.plot(andFunction[1:3],'ro',ms=10)

plt.plot(andFunction[0][1],andFunction[0][2],'bo',ms=10)
plt.ylim((-1,2))
plt.xlim((-1,2))
plt.show()


### you need to initialize the weights for your perceptron (I suggest
### using an np.array).

#Initialize andWeights to zero.
andWeights = np.zeros(3)


### A function for the input value of a perceptron.  It should compute
### the dot product of a weight vector and an input vector.

def inValue (weights, featureVals):
    #computing the dotProduct of weights and the feature values
    resVal = np.sum(weights[0] * featureVals[0] + weights[1] * featureVals[1] + weights[2] * featureVals[2])
    return resVal # Return the computed result value.

### You need to implement a step activation function here.  This
### function should return a 1 or 0 to match the class values defined
### in the andFunction above.

# Since the question asks threshold activation function, I have used this.
def stepActivation(z):
    # If z>0 return 1 else return 0
    if z>=0:
        return 1
    else:
        return 0 

# For activation we can also use sigmoid function so that the
# activation value lies between 0 and 1.
def sigmoidActivation(z):
    return 1.0/(1.0 + np.exp(-z)) # Use sigmoid function for calculation.

### Weight update goes here. This function should return an np.array
### with 3 components.

def stepUpdate(weights, featureVals, observedClass, learningRate):
    w1 = weights[0]
    w2 = weights[1]
    w3 = weights[2]
    dataLabel = featureVals[3]

    #Calculate the updated Value.
    netVal = inValue(weights, featureVals[0:3])
    #print("val=",netVal)

    # Calculate the value after StepActivation.
    actVal = stepActivation(netVal)

    # Calculate the weights and create a numpy_array
    w1 = w1 + learningRate*featureVals[0]*(dataLabel-actVal)
    w2 = w2 + learningRate*featureVals[1]*(dataLabel-actVal)
    w3 = w3 + learningRate*featureVals[2]*(dataLabel-actVal)
    newWeights = np.array([w1,w2,w3])
    return newWeights #Return the calculated weights.
   

### Suggested learning rate and epoch numbers.  

#Initialize values for learningRate and epoch.   
learningRate = 0.09
epoch = 50

# Set the value for observedClass
observedClass = andFunction[3]


### loop running training over the suggested number of epochs and an
### embedded loop over each data point. 

for i in range(epoch):
    # For every val, compute the stepUpdate, modify the andWeights accordingly till we converge to
    # a good value.
    for dataPoint in andFunction:
        #inputVals = andFunction[dataPoint]
        andWeights = stepUpdate(andWeights, dataPoint, observedClass, learningRate)

print andWeights
trainingData = andWeights # Initialize the training data equal to andWeights.
print "train=",trainingData
### after the loop above, your weights should be able to correctly
### classify the training data.  Plot the training data and the
### decision boundary that your trained weights yield.

# Now we will use training data to classify them.
# We will make use of graph to classify. There will be decision boundary that will be obtained from the graph plot.


# plotting

#print np.shape(trainingData)
for i in range(len(andFunction)):
    plt.plot(andFunction[1:3],'ro',ms=10)


plt.plot(andFunction[0][1],andFunction[0][2],'bo',ms=10)
plt.ylim((-1,2))
plt.xlim((-1,2))




#returns evenly spaced numbers over the interval
x = np.linspace(-1, 2, 1000)

print(np.shape(x))
#y = np.multiply(-(trainingData[0] / trainingData[1]),x)
#print(np.shape(y))
plt.plot(x,-(trainingData[1] / trainingData[2])*x -(trainingData[0] / trainingData[2]))
#print("val1=",trainingData[0] / trainingData[1])
#print("val2=",trainingData[2] / trainingData[1])
plt.title('andFunction')
plt.show()



