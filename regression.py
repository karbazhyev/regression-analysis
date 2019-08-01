# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 12:32:45 2017

@author: msahr
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
from sklearn.pipeline import Pipeline


def linear_regression( data , y , alpha , *test_data):
    
    temp = data.transpose() @ data
    temp = ( temp + temp.transpose() ) /2 
    beta = ( np.linalg.inv( temp + alpha * np.eye( np.size(temp , 0) ) ) ).dot( data.transpose().dot( y ) )
    
    train_y_estim = data.dot( beta )
    train_MSE = np.sqrt(np.mean( ( y - train_y_estim ) ** 2 ) )
    if test_data:
        test_x , test_y = test_data
        
        test_y_estim = test_x.dot( beta )
        test_MSE = np.sqrt( np.mean( ( test_y - test_y_estim ) ** 2 ) )
        #plt.plot(test_y)
        #plt.plot(test_y_estim)
        return beta, train_MSE , test_MSE , train_y_estim
    else:
        return beta, train_MSE , train_y_estim
                    
    

def k_fold_x_validation( x , y , k , mode , *param):
    
    idx_shuffle = np.random.permutation( np.size( x, 0 ) )
    if len( x.shape ) == 2:
        x = x[ idx_shuffle , :]
    else:
        x = x[ idx_shuffle ]
    y = y[ idx_shuffle ]
    
    test_size = int( np.size( x , 0 ) / k )
    test_MSE = np.zeros( ( k ) )
    for ii in range( k ):
        test_x = x[ ii * test_size : (ii + 1 )* test_size ]
        test_y = y[ ii * test_size : (ii + 1 )* test_size ]
        
        train_x = np.delete( x , np.arange( ii * test_size , ( ii + 1 ) * test_size ) , 0 )
        train_y = np.delete( y , np.arange( ii * test_size , ( ii + 1 ) * test_size ) )
        
        if mode == 'Linear':
            if param:
                alpha = param[0]
            else:
                alpha = 0.01
            beta , train_MSE , test_MSE[ ii ] , y_etim= linear_regression( train_x , train_y , alpha,  test_x , test_y )
        elif mode == 'RandomForest':
            if param:
                depth = param[0]
            else:
                depth = 4
                
            regressor = RandomForestRegressor( n_estimators = 20 , max_depth = depth )
            regressor.fit( train_x, train_y )
            test_y_estim = regressor.predict( test_x )
            test_MSE[ ii ] = np.sqrt( np.mean( ( test_y - test_y_estim ) ** 2 ) )
        elif mode == 'NeuralNetwork':
            if param:
                h_layer_sizes = param
                
            else:
                h_layer_sizes = (30)
            neural_net = MLPRegressor( hidden_layer_sizes = h_layer_sizes , solver='sgd')
            neural_net.fit( train_x, train_y )
            test_y_estim = neural_net.predict( test_x )
            test_MSE[ ii ] = np.sqrt( np.mean( ( test_y - test_y_estim ) ** 2 ) )
        elif mode == 'Polynomial':
            if param:
                deg = param[0]
            else:
                deg = 2
            
            poly = PolynomialFeatures(degree = deg)
            train_x_poly = poly.fit_transform(train_x)
            test_x_poly = poly.fit_transform(test_x)
            regressor = linear_model.Ridge (alpha = 0.01 )
            regressor.fit( train_x_poly , train_y )
            test_y_estim = regressor.predict( test_x_poly )
            test_MSE[ ii ] = np.sqrt( np.mean( ( test_y - test_y_estim ) ** 2 ) )
        elif mode == 'Lasso':
            if param:
                a = param[0]
            else:
                a = 0.1
            regressor = linear_model.Lasso(alpha = a)
            regressor.fit( train_x , train_y )
            test_y_estim = regressor.predict( test_x )
            test_MSE[ ii ] = np.sqrt( np.mean( ( test_y - test_y_estim ) ** 2 ) )
        elif mode == 'Ridge':
            if param:
                a = param[0]
                
            else:
                a = 0.1
            regressor = linear_model.Ridge (alpha = a )
            
            regressor.fit( train_x , train_y )
            test_y_estim = regressor.predict( test_x )
            test_MSE[ ii ] = np.sqrt( np.mean( ( test_y - test_y_estim ) ** 2 ) )
        
        else:
            raise ValueError('Regression mode used is noot defined!')
            
        
        
    if mode == 'Linear':
        beta , train_MSE, y_estim = linear_regression( x , y , alpha )
        
        return beta , train_MSE , test_MSE , y_estim
    
    elif mode == 'RandomForest':
        regressor = RandomForestRegressor( n_estimators = 20 , max_depth = depth)
        regressor.fit( x , y )
        y_estim = regressor.predict( x )
        train_MSE = np.sqrt( np.mean( ( y - y_estim ) ** 2 ) )
        
        plt.figure()
        plt.scatter( y_estim , y- y_estim , alpha = 1 , marker = 'o')
        plt.xlabel('Fitted Value', fontsize = 30)
        plt.ylabel('Residual', fontsize = 30)
        plt.xticks( fontsize = 30)
        plt.yticks( fontsize = 30)
        plt.figure()
        plt.scatter( y ,  y_estim , alpha = 1 , marker = 'o')
        plt.xlabel('Data', fontsize = 30)
        plt.ylabel('Fitted Value', fontsize = 30)
        plt.xticks( fontsize = 30)
        plt.yticks( fontsize = 30)
        
        return regressor , train_MSE , test_MSE , y_estim
    
    elif mode == 'NeuralNetwork':
        neural_net = MLPRegressor( hidden_layer_sizes = h_layer_sizes , solver='sgd' )
        neural_net.fit( x, y )
        y_estim = neural_net.predict( x )
        train_MSE = np.sqrt( np.mean( ( y - y_estim ) ** 2 ) )
                 
        
        
        
        return neural_net , train_MSE , test_MSE , y_estim
    elif mode == 'Polynomial':
        poly = PolynomialFeatures(degree = deg)
        x_poly = poly.fit_transform( x )
        regressor = linear_model.Ridge (alpha = 0.01 )
        regressor.fit( x_poly , y )
        y_estim = regressor.predict( x_poly )
        train_MSE = np.sqrt( np.mean( ( y - y_estim ) ** 2 ) )
        return regressor , train_MSE , test_MSE , y_estim
        
    elif mode == 'Lasso':
        regressor = linear_model.Lasso(alpha = a)
        regressor.fit( x , y )
        y_estim = regressor.predict( x )
        train_MSE = np.sqrt( np.mean( ( y - y_estim ) ** 2 ) )
        
        plt.figure()
        plt.scatter( y_estim , y- y_estim , alpha = 1 , marker = 'o')
        plt.xlabel('Fitted Value', fontsize = 30)
        plt.ylabel('Residual', fontsize = 30)
        plt.xticks( fontsize = 30)
        plt.yticks( fontsize = 30)
        plt.figure()
        plt.scatter( y ,  y_estim , alpha = 1 , marker = 'o')
        plt.xlabel('Data', fontsize = 30)
        plt.ylabel('Fitted Value', fontsize = 30)
        plt.xticks( fontsize = 30)
        plt.yticks( fontsize = 30)
        
        return regressor.coef_ , train_MSE , test_MSE , y_estim
    elif mode == 'Ridge':
        regressor = linear_model.Ridge (alpha = a )
        regressor.fit( x , y )
        y_estim = regressor.predict( x )
        train_MSE = np.sqrt( np.mean( ( y - y_estim ) ** 2 ) )
        
        plt.figure()
        plt.scatter( y_estim , y- y_estim , alpha = 1 , marker = 'o')
        plt.xlabel('Fitted Value', fontsize = 30)
        plt.ylabel('Residual', fontsize = 30)
        plt.xticks( fontsize = 30)
        plt.yticks( fontsize = 30)
        plt.figure()
        plt.scatter( y ,  y_estim , alpha = 1 , marker = 'o')
        plt.xlabel('Data', fontsize = 30)
        plt.ylabel('Fitted Value', fontsize = 30)
        plt.xticks( fontsize = 30)
        plt.yticks( fontsize = 30)
        
        return regressor.coef_ , train_MSE , test_MSE , y_estim