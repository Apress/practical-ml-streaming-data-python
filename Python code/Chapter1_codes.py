#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 20 19:22:37 2020

@author: Dr. Sayan Putatunda
"""

""" Chapter 1 Codes """

## Sliding window code
import itertools
from itertools import tee
from itertools import zip_longest as zip

def window(iterations, size):
    n = tee(iterations, size)
    for i in range(1, size):
        for each in n[i:]:
            next(each, None)
    return zip(*n)

y=range(10)
for each in window(y, 4):
    print(list(each))
   
##################################################################################



########### Mini-batch gradient descent for linear regresson code ###############   

import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
from pandas import read_csv

np.random.seed(111)

# creating data 
mean = np.array([6.0, 7.0]) 
covariance = np.array([[1.0, 0.94], [0.95, 1.2]]) 
df = np.random.multivariate_normal(mean, covariance, 500) 
  
df.shape
# output- (10000, 2)

df
#Output-
#array([[6.97833668, 8.30414776],
#       [4.66509294, 5.31806407],
#       [6.88804517, 7.71734893],
#       ...,
#       [5.75153443, 6.73145512],
#       [4.93357924, 6.72570148],
#       [7.31794626, 8.4076224 ]])

X = df[:,:-1]
Y = df[:,-1]
# A column with 1's is added
X_new = np.c_[np.ones((500, 1)), X] 

# Random initialization of the estimate "reg_coef"
np.random.seed(333)
reg_coef = np.random.randn(2,1)  

# here, size_batch_mini= minibatch size
# lr= learning rate
#  max_iters= number of batches used
lr=0.01
num = 100
max_iters = 100
size_batch_mini = 50


t0, t1 = 400, 1200
def lrs(step):
    return t0 / (step + t1)

reg_coef_all = []

step = 0
for j in range(max_iters):
    batches_index  = np.random.permutation(num)
    X_batches = X_new[batches_index]
    y_batches = Y[batches_index]
    for i in range(0, num, size_batch_mini):
        step += 1
        Yi = y_batches[i:i+size_batch_mini]
        Xi = X_batches[i:i+size_batch_mini]
        # compute the gradient
        gradient = 2/size_batch_mini * Xi.T.dot(Xi.dot(reg_coef) - Yi)
        lr = lrs(step)
        # update
        reg_coef = reg_coef - lr * gradient
        reg_coef_all.append(reg_coef)
 
# Output       
reg_coef
reg_coef_all

########################################################################################



    
############## Perform linear regression using Stochastic Gradient Descent
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error, r2_score

np.random.seed(111)

# creating the dataset
mean = np.array([6.0, 7.0]) 
covariance = np.array([[1.0, 0.94], [0.95, 1.2]]) 
df = np.random.multivariate_normal(mean, covariance, 10000) 
  
df.shape
# output- (10000, 2)

df
#Output-
#array([[6.97833668, 8.30414776],
#       [4.66509294, 5.31806407],
#       [6.88804517, 7.71734893],
#       ...,
#       [5.75153443, 6.73145512],
#       [4.93357924, 6.72570148],
#       [7.31794626, 8.4076224 ]])

X = df[:,:-1]
Y = df[:,-1]

## Split into train and test
train_X, test_X, train_y, test_y = train_test_split(X, Y, test_size=0.20, random_state=333)

### Use the SGD regressor
mod = SGDRegressor()

### Fit the model
mod.fit(train_X, train_y)
#SGDRegressor(alpha=0.0001, average=False, early_stopping=False, epsilon=0.1,
#             eta0=0.01, fit_intercept=True, l1_ratio=0.15,
#             learning_rate='invscaling', loss='squared_loss', max_iter=1000,
#             n_iter_no_change=5, penalty='l2', power_t=0.25, random_state=None,
#             shuffle=True, tol=0.001, validation_fraction=0.1, verbose=0,
#             warm_start=False)

### Print the coefficient and intercept values
print("Coefficients: \n", mod.coef_) 
#Coefficients: [0.95096598]

print("Intercept", mod.intercept_)
#Intercept [1.17712595]

### Predict on the test data
pred = mod.predict(test_X)

# calculating the prediction error
error = np.sum(np.abs(test_y - pred) / test_y.shape[0]) 
print("Mean absolute error (MAE) = ", error) 
# Mean absolute error (MAE) =  0.4493164833335055
############################################################################################


### Hyperplane generator using scikit-multiflow
from skmultiflow.data import HyperplaneGenerator
import pandas as pd
import numpy as np

create = HyperplaneGenerator(random_state = 888, n_features= 10, noise_percentage = 0)
create.prepare_for_use()
X , Y = create.next_sample(10000)
data = pd.DataFrame(np.hstack((X, np.array([Y]).T)))

data.shape 
# output- (10000, 11)  
print(data.head())    
# Output:
#         0         1         2         3   ...        7         8         9    10
#0  0.899478  0.248365  0.030172  0.072447  ...  0.633460  0.283253  0.365369  1.0
#1  0.705171  0.648509  0.040909  0.211732  ...  0.026095  0.446087  0.239105  0.0
#2  0.091587  0.977452  0.411501  0.458305  ...  0.181444  0.303406  0.174454  0.0
#3  0.635272  0.496203  0.014126  0.627222  ...  0.517752  0.570683  0.546333  1.0
#4  0.450078  0.876507  0.537356  0.495684  ...  0.606895  0.217841  0.912944  1.0
#
#[5 rows x 11 columns] 

# Store it in csv                   
data.to_csv('data_stream_hyperplane.csv', index=False)
#####################################################################################


### Agarwal generator using scikit-multiflow
from skmultiflow.data import AGRAWALGenerator
import pandas as pd
import numpy as np

create = AGRAWALGenerator(random_state=333)
create.prepare_for_use()
X , Y = create.next_sample(10000)
data = pd.DataFrame(np.hstack((X, np.array([Y]).T)))

data.shape 
# output- (1000, 10)  
print(data.head())    
# Output:
#               0             1     2  ...     7              8    9
#0   90627.841313      0.000000  33.0  ...  20.0   24151.832875  0.0
#1   33588.924462  17307.813671  72.0  ...  29.0  315025.363876  0.0
#2   24375.065287  12426.917711  39.0  ...   4.0  363158.576720  0.0
#3   82949.727691      0.000000  68.0  ...   2.0   35758.528073  0.0
#4  149423.790417      0.000000  52.0  ...  29.0   98440.362484  1.0
#
#[5 rows x 10 columns]

# Store it in csv                   
data.to_csv('data_stream_agarwal.csv', index=False)
#####################################################################################










### Create stream from a CSV file
# Import
from skmultiflow.data.file_stream import FileStream

# Setup the data stream
data_stream= FileStream('./Absenteeism_at_work.csv')


# Retrieving one sample
data_stream.next_sample()
# Output-
#(array([[ 11.   ,  26.   ,   7.   ,   3.   ,   1.   , 289.   ,  36.   ,
#          13.   ,  33.   , 239.554,  97.   ,   0.   ,   1.   ,   2.   ,
#           1.   ,   0.   ,   1.   ,  90.   , 172.   ,  30.   ]]), array([4]))

 # Retrieving 5 samples
data_stream.next_sample(5)
# Output-
#(array([[ 36.   ,   0.   ,   7.   ,   3.   ,   1.   , 118.   ,  13.   ,
#          18.   ,  50.   , 239.554,  97.   ,   1.   ,   1.   ,   1.   ,
#           1.   ,   0.   ,   0.   ,  98.   , 178.   ,  31.   ],
#        [  3.   ,  23.   ,   7.   ,   4.   ,   1.   , 179.   ,  51.   ,
#          18.   ,  38.   , 239.554,  97.   ,   0.   ,   1.   ,   0.   ,
#           1.   ,   0.   ,   0.   ,  89.   , 170.   ,  31.   ],
#        [  7.   ,   7.   ,   7.   ,   5.   ,   1.   , 279.   ,   5.   ,
#          14.   ,  39.   , 239.554,  97.   ,   0.   ,   1.   ,   2.   ,
#           1.   ,   1.   ,   0.   ,  68.   , 168.   ,  24.   ],
#        [ 11.   ,  23.   ,   7.   ,   5.   ,   1.   , 289.   ,  36.   ,
#          13.   ,  33.   , 239.554,  97.   ,   0.   ,   1.   ,   2.   ,
#           1.   ,   0.   ,   1.   ,  90.   , 172.   ,  30.   ],
#        [  3.   ,  23.   ,   7.   ,   6.   ,   1.   , 179.   ,  51.   ,
#          18.   ,  38.   , 239.554,  97.   ,   0.   ,   1.   ,   0.   ,
#           1.   ,   0.   ,   0.   ,  89.   , 170.   ,  31.   ]]),
# array([0, 2, 4, 2, 2]))

data_stream.has_more_samples()
# Output-
# True

data_stream.n_remaining_samples()
# Output-
# 734

#####################################################################################
