# -*- coding: utf-8 -*-
"""
Created on Tue Dec 18 14:34:50 2018

@author: Aleksandar Haber

Using MultiLayer Perceptron (MLP) to identify a discrete-time ARX model of the system:

\dot{x}(t)=A*x(t)+Bu(t)
y(t)=C*x(t)

where A,B,C are the system matrices
x(t) is the state at the time instant t
y(t) is the output (our observations)
u(t) is the external input

The network "estimates" the ARX model 

y_{k+1}=\sum_{i=0}^{p} Y_{i} y_{k-p+i} + \sum_{i=0}^{p} U_{i} u_{k-p+i}

where 
Y_{i},U_{i} are constant matrices (scalars in our case)
y_{i},u_{i} are the outputs and inputs at the discrete-time instant i
p is the past horizon

"""

# uncomment these two lines if you installed this package:
# https://github.com/plaidml/plaidml
# this package is used to perform GPU computations on almost any GPU...

# import plaidml.keras
# plaidml.keras.install_backend()

import numpy as np
import matplotlib.pyplot as plt

###############################################################################
#                   Model definition
###############################################################################

# First, we need to define the system matrices of the state-space model:
# this is a continuous-time model, we will simulate it using the backward Euler method
A=np.matrix([[0, 1],[- 0.1, -0.000001]])
B=np.matrix([[0],[1]])
C=np.matrix([[1, 0]])

#define the number of time samples used for simulation and the discretization step (sampling)
time=400
sampling=0.5

# this is the past horizon
past=2
###############################################################################
#        This function formats the input and output data
###############################################################################
def form_data(input_seq, output_seq,past):
    data_len=np.max(input_seq.shape)
    X=np.zeros(shape=(data_len-past,2*past))
    Y=np.zeros(shape=(data_len-past,))
    for i in range(0,data_len-past):
        X[i,0:past]=input_seq[i:i+past,0]
        X[i,past:]=output_seq[i:i+past,0]
        Y[i]=output_seq[i+past,0]
    return X,Y
###############################################################################
#                  Create the training data
###############################################################################
#define an input sequence for the simulation
input_seq_train=np.random.rand(time,1)
#define an initial state for simulation
x0_train=np.random.rand(2,1)
# here we simulate the dynamics
from backward_euler import simulate
state_seq_train,output_seq_train=simulate(A,B,C,x0_train,input_seq_train, time ,sampling)  
output_seq_train=output_seq_train.T
output_seq_train=output_seq_train[0:-1]
X_train,Y_train= form_data(input_seq_train, output_seq_train, past)
###############################################################################
#                  Create the validation data
###############################################################################
#define an input sequence for the simulation
input_seq_validate=np.random.rand(time,1)
#define an initial state for simulation
x0_validate=np.random.rand(2,1)
state_seq_validate,output_seq_validate=simulate(A,B,C,x0_validate,input_seq_validate, time ,sampling)  
output_seq_validate=output_seq_validate.T
output_seq_validate=output_seq_validate[0:-1]
X_validate,Y_validate= form_data(input_seq_validate, output_seq_validate, past)
###############################################################################
#                  Create the test data
###############################################################################
#define an input sequence for the simulation
input_seq_test=np.random.rand(time,1)
#define an initial state for simulation
x0_test=np.random.rand(2,1)
state_seq_test,output_seq_test=simulate(A,B,C,x0_test,input_seq_test, time ,sampling)  
output_seq_test=output_seq_test.T
output_seq_test=output_seq_test[0:-1]
X_test,Y_test= form_data(input_seq_test, output_seq_test, past)
###############################################################################
#               Create the MLP network and train it
###############################################################################
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
#model.add(Dense(2, activation='relu',use_bias=False, input_dim=2*past))
model.add(Dense(2, activation='linear',use_bias=False, input_dim=2*past))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
history=model.fit(X_train, Y_train, epochs=1000, batch_size=20, validation_data=(X_validate,Y_validate), verbose=2)

###############################################################################
#   use the test data to investigate the prediction performance
###############################################################################
network_prediction = model.predict(X_test)
from numpy import linalg as LA
Y_test=np.reshape(Y_test, (Y_test.shape[0],1))
error=network_prediction-Y_test

# this is the measure of the prediction performance in percents
error_percentage=LA.norm(error,2)/LA.norm(Y_test,2)*100

plt.figure()
plt.plot(Y_test, 'b', label='Real output')
plt.plot(network_prediction,'r', label='Predicted output')
plt.xlabel('Discrete time steps')
plt.ylabel('Output')
plt.legend()
plt.savefig('prediction_offline.png')
#plt.show()

###############################################################################
#       plot training and validation curves
###############################################################################

loss=history.history['loss']
val_loss=history.history['val_loss']
epochs=range(1,len(loss)+1)
plt.figure()
plt.plot(epochs, loss,'b', label='Training loss')
plt.plot(epochs, val_loss,'r', label='Validation loss')
plt.title('Training and validation losses')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.xscale('log')
#plt.yscale('log')
plt.legend()
plt.savefig('loss_curves.png')
#plt.show()
###############################################################################
#  do prediction on the basis of the past predicted outputs- this is an off-line mode
###############################################################################

# for the time instants from 0 to past-1, we use the on-line data 

predict_time=X_test.shape[0]-2*past

Y_predicted_offline=np.zeros(shape=(predict_time,1))
Y_past=network_prediction[0:past,:].T
X_predict_offline=np.zeros(shape=(1,2*past))

for i in range(0,predict_time):
    X_predict_offline[:,0:past]=X_test[i+2*past,0:past]
    X_predict_offline[:,past:2*past]=Y_past    
    y_predict_tmp= model.predict(X_predict_offline)
    Y_predicted_offline[i]=y_predict_tmp
    Y_past[:,0:past-1]=Y_past[:,1:]
    Y_past[:,-1]=y_predict_tmp

error_offline=Y_predicted_offline-Y_test[past:-past,:]
error_offline_percentage=LA.norm(error_offline,2)/LA.norm(Y_test,2)*100

#plot the offline prediction and the real output
plt.plot(Y_test[past:-past,:],'b',label='Real output')
plt.plot(Y_predicted_offline, 'r', label='Offline prediction')
plt.xlabel('Discrete time steps')
plt.ylabel('Output')
plt.legend()    
plt.savefig('prediction_offline.png')
#plt.show()

plt.figure()
#plot the absolute error (offline and online)
plt.plot(abs(error_offline),'r',label='Offline error')
plt.plot(abs(error),'b',label='Online error')
plt.xlabel('Discrete time steps')
plt.ylabel('Absolute prediction error')
plt.yscale('log')
plt.legend()   
plt.savefig('errors.png')
#plt.show()








  