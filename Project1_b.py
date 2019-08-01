# -*- coding: utf-8 -*-
"""
Created on Sat Jan 28 23:34:45 2017

@author: msahr
"""

import numpy as np
import matplotlib.pyplot as plt
from csvtoarray import*
from daya_total import*
from feature import*
from regression import*


data = csvtoarray2( 'housing_data.csv' )
train_MSE = np.zeros(13)
test_RMSE = np.zeros(13)
test_MSE_std = np.zeros(13)
data = np.c_[np.ones(506) , data]
for ii in range(1, 14):
    beta , train_MSE, test_MSE, y_estim = k_fold_x_validation( data[:,[ii, 0] ], data[:,-1] , 10 , 'Linear' , 0.001)
    test_RMSE[ii-1] = np.mean(test_MSE)
    test_MSE_std[ii-1] = np.std(test_MSE)
    plt.errorbar(ii-1 , test_RMSE[ii-1] , test_MSE_std[ii-1], lw=3)


plt.xticks( range(13) , ['CRIM' , 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE' , 'DIS' , 'RAd' , 'TAx' ,'PTRatio', 'B', 'LSTAT'], fontsize = 32 )
plt.yticks( fontsize = 32 )
plt.xlabel('feature', fontsize = 35)
plt.ylabel('RMSE and STD of mean squared error for test data in 10 fold cross validation for each feature', fontsize = 25)
plt.xlim(-0.5, 12.5)

plt.figure()

beta , train_MSE, test_MSE , y_estim = k_fold_x_validation( data[:,0:-1 ]/np.max(data[:,0:-1] , axis = 0), data[:,-1] , 10 , 'NeuralNetwork' , (25))
print( np.mean(test_MSE) , np.std(test_MSE))

print('salam')
train_MSE = np.zeros(5)
test_RMSE = np.zeros(5)
test_std = np.zeros(5)

for ii in range(5):
    beta , train_MSE[ii], test_MSE, y_estim = k_fold_x_validation( data[:,0:-1 ]/np.max(data[:,0:-1] ,
                    axis = 0), data[:,-1] , 10 , 'Polynomial' , ii+1 )
    test_RMSE[ii] = np.mean(test_MSE)
    test_std[ii] = np.std(test_MSE)
    
plt.figure()
plt.errorbar(range(1,6) , test_RMSE , test_std, lw=3)
plt.xticks( fontsize = 32 )
plt.yticks( fontsize = 32 )
plt.xlabel('Degree', fontsize = 35)
plt.ylabel('RMSE and STD of mean squared error for test data in 10 fold cross validation', fontsize = 25)
plt.xlim(0.5, 5.5)

alpha = [0.0001, 0.001, 0.01, 0.1, 1 , 10 , 0.00]
train_MSE = np.zeros(7)
test_RMSE = np.zeros(7)
test_std = np.zeros(7)
for ii in range(7):
    beta , train_MSE[ii], test_MSE, y_estim = k_fold_x_validation( data[:,0:-1 ] ,
                     data[:,-1] , 10 , 'Ridge' , alpha[ii] )
    test_RMSE[ii] = np.mean(test_MSE)
    test_std[ii] = np.std(test_MSE)
    
plt.figure()
plt.errorbar(alpha , test_RMSE , test_std, lw=3)
plt.xticks( fontsize = 32 )
plt.yticks( fontsize = 32 )
plt.xlabel('Alpha', fontsize = 35)
plt.ylabel('Normalized RMSE and STD of mean squared error for test data in 10 fold cross validation', fontsize = 25)
plt.xlim(0.00002, 200)