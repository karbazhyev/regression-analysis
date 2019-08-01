# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 01:26:13 2017

@author: Mojtaba
"""

import numpy as np



def string_to_feature( data , options , mode = 'remove' ):
    total_week_num = 15
    total_day_num = 7
    total_start_time_num = 6
    total_work_flow_num = 5
    total_file_num = 30
    total_backup_time = 5
    
    
    
    if 'week' in options:
        week_dic = { i+1 : feature_vec( i , total_week_num ) for i in range( total_week_num) }
    elif mode == 'remove':
        week_dic = { i+1 : [] for i in range( total_week_num )}
    else:
        week_dic = { i+1 : i+1 for i in range( total_week_num )}
        
    if 'day' in options:
        day_dic = { i : feature_vec( i , total_day_num ) for i in range( total_day_num ) }
    elif mode == 'remove':
        day_dic = { i : [] for i in range( total_week_num) }
    else:
        day_dic = { i : i for i in range( total_week_num) }
        
    if 'start_time' in options:
        start_time_dic = { 4*(i) + 1 : feature_vec( i , total_start_time_num )for i in range (total_start_time_num) }
    elif mode == 'remove':
        start_time_dic = { 4*(i) + 1 : [] for i in range (total_start_time_num) }
    else:
        start_time_dic = { 4*(i) + 1 : 4*(i+1) + 1 for i in range (total_start_time_num) }
        
    if 'work_flow' in options:
        work_flow_dic = { i : feature_vec( i , total_work_flow_num ) for i in range( total_work_flow_num ) }
    else:
        work_flow_dic = { i : i for i in range( total_work_flow_num ) }
        
    if 'file_name' in options:
        file_name_dic = { i : feature_vec( i , total_file_num ) for i in range( total_file_num ) }
    else:
        file_name_dic = { i : i for i in range( total_file_num ) }
        
    if 'backup_time' in options:
        backup_time_dic = { i : feature_vec( i , total_backup_time ) for i in range( total_backup_time ) }
    else:
        backup_time_dic = { i : i for i in range( total_backup_time ) }
        
    for i in range( np.size( data , 0 ) ):
        temp = np.hstack( ( 1, week_dic[ data[ i , 0 ] ] , day_dic[ data[ i , 1 ] ] ,
                     start_time_dic[ data[ i , 2 ] ] , work_flow_dic[ data[ i , 3 ] ] ,
                     file_name_dic[ data[ i , 4 ] ] , backup_time_dic[ data[ i , 5 ] ] ) ) 
        if i == 0:
            data_feature = np.zeros(( np.size( data , 0 ) , temp.size ) )
        
        data_feature[ i , : ] = temp
        
           
            
    return data_feature
        
         
        
def feature_vec( i , total ):
    out = np.zeros( ( total ) )
    out[ i ] = 1
    return out