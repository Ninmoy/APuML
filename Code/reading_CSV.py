# -*- coding: utf-8 -*-
"""
Created on Sat Sep 19 19:20:56 2020

@author: NINMOYpc
"""

import csv 

# opening the CSV file 
with open('for_prediction.csv', mode ='r')as file: 	
    # reading the CSV file 
    csvFile = csv.reader(file) 

# displaying the contents of the CSV file 
    for lines in csvFile: 
        print(lines) 
