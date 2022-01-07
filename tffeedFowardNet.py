from sys import dont_write_bytecode

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib as plt

mnist = keras.datasets.mnist

##(balaTrain, sadiqTrain), (balaTest, sadiqTest) = mnist.load_data()

xTrain = np.loadtxt('train_X.csv', delimiter = ',').reshape(-1,28,28)
yTrain = np.loadtxt('train_label.csv', delimiter = ',')
yTrain = yTrain.argmax(axis=1)

xTest = np.loadtxt('test_X.csv', delimiter = ',').reshape(-1,28,28)
yTest = np.loadtxt('test_label.csv', delimiter = ',')
yTest = yTest.argmax(axis=1)

#normalize data
xTrain , xTest = xTrain/255.0, xTest/255.0


model = keras.models.Sequential([
    keras.layers.Flatten(input_shape = (28,28)),
    keras.layers.Dense(1000, activation = 'tanh'), #number is the output sizes
    keras.layers.Dense(1000, activation = 'relu'), #number is the output sizes
    keras.layers.Dense(10)   
])

print(model.summary())

# loss and optimizer 
loss = keras.losses.SparseCategoricalCrossentropy(from_logits = True)
optim = keras.optimizers.Nadam(lr = 0.001)
metrics = ['accuracy']

model.compile(loss = loss, optimizer = optim, metrics =  metrics)

#training
batch_size = 20
epochs = 7

model.fit(xTrain, yTrain, batch_size = batch_size, epochs = epochs, shuffle = True, verbose = 2)


#evaluate model
print('Bala')

model.evaluate(xTest, yTest, batch_size = batch_size, verbose=2 )

#predictions
probability_model = keras.models.Sequential([
    model,
    keras.layers.Softmax()    
])


predictions = probability_model(xTest)
pred0 = predictions[0]
print(pred0)
label0 = np.argmax(pred0)
print(label0)

#model + softmax

predictions= model(xTest)
predictions = tf.nn.softmax(predictions)
pred0 = predictions[0]
print(pred0)
label0 = np.argmax(pred0)
print(label0)


pred05s = predictions[0:5]
print(pred05s.shape)
label05s = np.argmax(pred05s, axis=1)
print(label05s)