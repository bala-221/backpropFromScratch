# -*- coding: utf-8 -*-
"""
Created on Tue Dec 27 02:49:50 2021

@author: babubaka

building NN from scrcatch, making it three layers
"""


import numpy as np
import matplotlib.pyplot as plt
import random

xTrain = np.loadtxt('train_X.csv', delimiter = ',').T
yTrain = np.loadtxt('train_label.csv', delimiter = ',').T

xTest = np.loadtxt('test_X.csv', delimiter = ',').T
yTest = np.loadtxt('test_label.csv', delimiter = ',').T

class NeuralNetworkScratch:    
    
    def __init__(self, learnRate, maxIter, nH):
        self.learnRate = learnRate
        self.maxIters = maxIter
        self.nH = nH
    

    #index = random.randrange(0, xTrain.shape[1])
    #plt.imshow(xTrain[:, index].reshape(28,28), cmap = 'gray')
    #plt.show()

    def _tanh(self,x):
        return np.tanh(x)


    def _relu(self,x):
      return np.maximum(x, 0)

    def _softmax(self,x):    
     expEx = np.exp(x)     
     return expEx/np.sum(expEx, axis = 0)

    def _derivativeTanh(self, x):
     return  (1-np.tanh(x)**2)

    def _derivativeRelu(self, x):
     return np.array(x > 0, dtype = np.float32)


    
    def _initilizeParams(self, nX, nH, nY):
        
        w1 = np.random.randn(nH, nX) * 0.01
        b1 = np.zeros((nH,1))
        
        
        w2 = np.random.randn(nH, nH) * 0.01
        b2 = np.zeros((nH,1))
        
        w3 = np.random.randn(nY, nH) * 0.01
        b3 = np.zeros((nY,1))
        
        parameters = {
            'w1' : w1,
            'b1' : b1,
            'w2' : w2,
            'b2' : b2,
            'w3' : w3,
            'b3' : b3,
            }
        return parameters


    #forward propagation:

    def _forwardPropagation(self, x, parameters):            
        w1 = parameters['w1']
        b1 = parameters['b1']
        w2 = parameters['w2']
        b2 = parameters['b2']        
        w3 = parameters['w3']
        b3 = parameters['b3']
        
        z1 = w1 @ x  + b1
        a1 = self._relu(z1)
        
        z2 = w2 @ a1 + b2
        a2 = self._tanh(z2)
        
        z3 = w3 @ a2 + b3
        a3 = self._softmax(z3)
        
        
        forwardCache = {
            'z1': z1,
            'a1': a1,        
            'z2': z2,
            'a2' : a2,
            'a3' : a3,
            }
        
        return forwardCache


    def _costFunction(self, a2, y):    
     m = y.shape[1] 
     cost = -(1/m)*np.sum(y*np.log(a2))
     return cost
     
    def _backPropagation(self, x, y, parameters, forwardCache):        
    
        w2 = parameters['w2']   
        w3 = parameters['w3']           
        
        a1 = forwardCache['a1']   
        a2 = forwardCache['a2']
        a3 = forwardCache['a3']
        
        m = x.shape[1]
        
        dz3 = a3 - y
        dw3 = (1/m)*(dz3 @ a2.T)
        db3 = (1/m) * np.sum(dz3, axis = 1, keepdims = True)
        
        dz2 = (w3.T @ dz3 ) * self._derivativeRelu(a2) 
        dw2 = (1/m)*(dz2 @ a1.T)
        db2 = (1/m) * np.sum(dz2, axis = 1, keepdims = True)
        
        dz1 = (1/m) * (w2.T @ dz2) * self._derivativeRelu(a1)    
        dw1 = (1/m) * dz1 @ x.T
        db1 = (1/m) * np.sum(dz1, axis = 1, keepdims = True)
        
        gradients = {
            'dw1': dw1,
            'db1': db1,
            'dw2': dw2,
            'db2': db2,
            'dw3': dw3,
            'db3': db3,}
        
        return gradients

    def _updateParameters(self, parameters, gradients, learnRate):    
        w1 = parameters['w1']
        b1 = parameters['b1']
        w2 = parameters['w2']
        b2 = parameters['b2']
        w3 = parameters['w3']
        b3 = parameters['b3']
        
        dw1 = gradients['dw1']
        dw2 = gradients['dw2']
        dw3 = gradients['dw3']
        db1 = gradients['db1']
        db2 = gradients['db2']
        db3 = gradients['db3']
       
        
        w1 = w1 - learnRate*dw1
        w2 = w2 - learnRate*dw2
        w3 = w3 - learnRate*dw3
        
        
        b1 = b1 - learnRate*db1
        b2 = b2 - learnRate*db2
        b3 = b3 - learnRate*db3
        
        parameters = {
            'w1' : w1,
            'b1' : b1,
            'w2' : w2,
            'b2' : b2,
            'w3' : w3,
            'b3' : b3,            
            }
        return parameters
        
        
        
        

    def fit(self, x, y):        
        nX = x.shape[0] #number of features
        nY = y.shape[0] #number of output neurons        
        nH = self.nH
        parameters = self._initilizeParams(nX, nH, nY)
        costList = np.zeros((self.maxIters,1))
        for i in range(self.maxIters):        
            forwardCache = self._forwardPropagation(x, parameters)        
            cost = self._costFunction(forwardCache['a3'], y)        
            gradients = self._backPropagation(x, y, parameters, forwardCache)        
            parameters = self._updateParameters(parameters, gradients, self.learnRate)
            costList[i] = cost
            
            if(i%(self.maxIters/10) == 0):
                print("Cost after", i, "iterations is ", cost)
        
        return parameters, costList
    
    
    def predict(self, x):
        forwardCache = self._forwardPropagation(x, parameters) 
        aOut = forwardCache['a3']
        aOut = np.argmax(aOut, 0)        
        return aOut
       
        #pass
    
    def accuracy(self, aOut, labels):    
        #forwardCache = self._forwardPropagation(inp, parameters)        
        #aOut = np.argmax(aOut, 0)
        yOut = np.argmax(labels, 0)
        acc = np.mean(aOut == yOut) * 100
        return acc        
       
        

maxIter = 100
nH = 500
learnRate = 0.1

#create the class
neuralNet = NeuralNetworkScratch(learnRate, maxIter, nH)

parameters, costList = neuralNet.fit(xTrain, yTrain)
testPredict = neuralNet.predict(xTest)
trainPredict = neuralNet.predict(xTrain)
accuracyTest = neuralNet.accuracy(aOut=testPredict, labels=yTest)
accuracyTrain = neuralNet.accuracy(aOut=trainPredict, labels=yTrain)
habu = 2
#parameters, costList = model(xTrain, yTrain, nH = nH, learnRate = learnRate, \
#iterations = iterations)

 

#train accuracy 
#print('Accuracy of Train dataset is:', accuracy(xTrain, yTrain, parameters), '%')
#print('Accuracy of Test dataset is:', accuracy(xTest, yTest, parameters), '%')

index = random.randrange(0, xTest.shape[1])
plt.imshow(xTest[:, index].reshape(28,28), cmap = 'gray')
plt.show()

#forwardCache = forwardPropagation(xTest[:, index].reshape(xTest.shape[0], 1), parameters)
#aOut = forwardCache['a2']
#aOut = np.argmax(aOut, 0)

#print('Our model says it is:', aOut[0])