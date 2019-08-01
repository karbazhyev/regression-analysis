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

for ii in range(5):
    
    day_total_data = day_total( data , [ii] , range(30) )
    
    my_xticks_dic = {0:'Mon',1:'Tue',2:'Wed',3:'Thur' , 4:'Fri' , 5:'Sat' , 6:'Sun'}
    my_xticks = [my_xticks_dic[ i % 7] for i in range(21)]
    x = range(3,6*21 , 6)
    #plt.xticks( x, my_xticks , fontsize = 30)
    #plt.plot( day_total_data[0:6*21,5] , label = 'workflow' + str(ii) ,linewidth=3.0 )
    
day_total_data = day_total( data , range(5) , range(30) )
#plt.plot( day_total_data[0:6*21,5] , label = 'Total'  ,linewidth=3.0)
#plt.legend(loc = 'upper left' , fontsize = 32)


backup_size = data[ : , 5 ]
data = np.delete (data , 5 , 1)
data_feature = string_to_feature( data , [    'day' , 'start_time' , 'work_flow' , 'file_name' ] , 'a'  )
hls = [ (5) , (10) , (15), (25)  , (40) , (70) , (100), (150), (200) , (300) , (500), (800)]
RMSE_sgd = np.zeros(len(hls))
Test_std_sgd = np.zeros(len(hls))
MSE_sgd = np.zeros(len(hls))

#RMSE_lbfgs = np.zeros(len(hls))
#Test_std_lbfgs = np.zeros(len(hls))
#MSE_lbfgs = np.zeros(len(hls))
for i in range(len(hls)):
    beta , train_MSE, test_MSE , y_estim = k_fold_x_validation( data[:,0:6] , backup_size , 10 , 'NeuralNetwork' , hls[i] )
    print(train_MSE, np.mean(test_MSE) , np.std(test_MSE) , test_MSE)
    RMSE_sgd[i] = np.mean(test_MSE)
    Test_std_sgd[i] = np.std(test_MSE)
    MSE_sgd[i] = train_MSE
    
plt.errorbar(hls ,RMSE_sgd,Test_std_sgd, label='sgd')
plt.errorbar(hls[0:-1] ,RMSE_lbfgs[0:-1], Test_std_lbfgs[0:-1], label='lbfgs')
plt.xlabel('Hidden Layer Size', fontsize = 30)
plt.ylabel('RMSE', fontsize = 30)
#numpy.savetxt('beta1.txt', beta, fmt='%.3f   '  )