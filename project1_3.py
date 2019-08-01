# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 12:05:56 2017

@author: msahr
"""


import numpy as np
import matplotlib.pyplot as plt
from csvtoarray import*
from daya_total import*
from feature import*
from regression import*


data = csvtoarray( 'network_backup_dataset.csv' )


#data_feature = np.delete( data_feature , [0] , 1)
data = np.c_[ data , data[:,0] *24*7 + data[:,1]* 24 + data[:, 2] ]
data = data[ np.argsort( data[ : , -1 ] )  ]
mysum = 0



backup_size = data[ : , 5 ]
data = np.delete (data , 5 , 1)
data_feature = string_to_feature( data , [    'day' , 'start_time' , 'work_flow' , 'file_name' ] , 'a'  )
for ii in range(5):
    idx = (data[:, 3] == ii)
    data_workflow = data[ idx , :]
    backup_size_workflow = backup_size[idx]
    beta , train_MSE, test_MSE = k_fold_x_validation( data_workflow[:,0:6] , backup_size_workflow , 10 , 'Ridge' , 0 )
    print(train_MSE, np.mean(test_MSE) , np.std(test_MSE) , np.mean(backup_size_workflow) , np.std(backup_size_workflow))
    plt.errorbar(ii ,np.mean(test_MSE)/np.mean(backup_size_workflow), np.std(test_MSE)/np.mean(backup_size_workflow) , lw = 3)
    plt.xlim(-0.5 , 4.5)
    plt.xticks( fontsize = 32 )
    plt.yticks( fontsize = 32 )
    plt.xlabel('Work Flow', fontsize = 35)
    plt.ylabel('Normalized RMSE and STD of error in cross validation', fontsize = 25)
 
train_MSE = np.zeros(6)
test_RMSE = np.zeros(6)
test_std = np.zeros(6)

for ii in range(1,7):
    beta , train_MSE[ii-1], test_MSE = k_fold_x_validation( data[:,0:6] , backup_size , 10 , 'Polynomial' , ii )
    test_RMSE[ii-1] = np.mean(test_MSE)
    test_std[ii-1] = np.std(test_MSE)
    
plt.figure()
plt.errorbar(range(1,7) , test_RMSE , test_std, lw=3)
plt.xticks( fontsize = 32 )
plt.yticks( fontsize = 32 )
plt.xlabel('Degree', fontsize = 35)
plt.ylabel('RMSE and STD of mean squared error for test data in 10 fold cross validation', fontsize = 25)
#numpy.savetxt('beta1.txt', beta, fmt='%.3f   '  )