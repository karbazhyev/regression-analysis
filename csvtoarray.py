# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 13:29:55 2017

@author: msahr
"""

import numpy as np
import csv
import re

def csvtoarray(csv_name):
    
     with open(csv_name , 'rt') as csvfile:
         reader = csv.reader(csvfile)
         data_string = list(reader)
         
     data_string = data_string[1:]
     weekdays = {'Monday' : '0' , 'Tuesday' : '1' , 'Wednesday' : '2' , 'Thursday': '3' ,
                 'Friday' : '4' , 'Saturday' : '5', 'Sunday' : '6'}
     
     for row in data_string:
         row[1] = weekdays[row[1]]
         row[3] = re.sub("[^0-9]", "", row[3])
         row[4] = re.sub("[^0-9]", "", row[4])
         
     data = np.array( [float(x) for x in data_string[0]] )
     for row in data_string[1:]:
         current_data = np.array( [float(x) for x in row] )
         data = np.vstack( (data, current_data) )
             
     return data
 
    
def csvtoarray2( csv_name ):

    with open( csv_name , 'rt') as csvfile:
        reader = csv.reader(csvfile)
        data_string = list(reader)
        
        
    data = np.zeros( ( 506 , 14 ) )       
    for i in range( len( data_string ) ):
        
        data[ i , : ] = np.array( [float(x) for x in data_string[ i ] ] )
        
    return data