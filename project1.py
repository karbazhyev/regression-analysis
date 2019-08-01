# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 12:05:56 2017

@author: msahr
"""


import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
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
    plt.xticks( x, my_xticks , fontsize = 30)
    plt.plot( day_total_data[0:6*21,5] , label = 'workflow' + str(ii) ,linewidth=3.0 )
    
day_total_data = day_total( data , range(5) , range(30) )
plt.plot( day_total_data[0:6*21,5] , label = 'Total'  ,linewidth=3.0)
plt.legend(loc = 'upper left' , fontsize = 32)


backup_size = data[ : , 5 ]
data = np.delete (data , 5 , 1)
times = data[:,6]
data[:,6] = 1
data_feature = string_to_feature( data , [     'day' , 'start_time' , 'work_flow', 'backup_time' ] , 'remove'  )




beta , train_MSE, test_MSE , y_estim = k_fold_x_validation( data_feature , backup_size , 10 , 'Linear' , 0.0001 )
print(train_MSE, np.mean(test_MSE) , np.std(test_MSE) , test_MSE)
y_estim_file = np.zeros(np.size(times))
file_ids = [3, 13 , 29]
for ii in range(len(file_ids)):
   
    idx = data[:,4] == file_ids[ii]
    y_estim_file= y_estim[idx]
    y_file = backup_size[idx]
    
    plt.subplot(3, 1, ii+1)
    plt.title('File' + str(file_ids[ii]) , fontsize = 30)
    plt.plot(y_file[0:6*21],label= ' Actual Value')
    plt.plot(y_estim_file[0:6*21], label = 'Fitted Value')
    plt.xticks( x, my_xticks , fontsize = 30)
    plt.legend(fontsize = 25)
    
    
    
    
results = sm.OLS(backup_size, data).fit_regularized(alpha=0.01, L1_wt=0.1)
print(results.summary())

#numpy.savetxt('beta1.txt', beta, fmt='%.3f   '  )