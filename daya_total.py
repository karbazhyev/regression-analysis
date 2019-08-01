# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 22:01:12 2017

@author: Mojtaba
"""
import numpy as np
def day_total(data, work_flow , file_id = range( 30 ) ):
    times = set( data[ : , -1 ] )
    file_idx = np.zeros( np.size( data, 0 ) )
    for ii in file_id:
        file_idx = np.logical_or( file_idx , data[:,4] == ii )
     
    data = data[file_idx , :]
    
    work_flow_idx = np.zeros( np.size(data, 0) )
    for jj in work_flow:
        work_flow_idx = np.logical_or( work_flow_idx , data[ : , 3 ] == jj )
        
    data = data[ work_flow_idx , : ]
     
    
    
    data_out = np.zeros( ( len(times), np.size(data,1) ) )
    
    ii = 0
    for jj in sorted( list( times ) ):
        times_idx = ( data[ :, -1 ] == jj )
        if np.any( times_idx ):
            temp = data[ times_idx ]
            data_out[ ii , : ] = temp[ 0 , : ]
            data_out[ ii , [ 5 , 6 ] ] = np.sum( temp[ : , [ 5 , 6 ] ] , 0 )
        ii += 1
        
    
    return data_out
        