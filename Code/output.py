# -*- coding: utf-8 -*-
"""
Created on Sat Sep 19 20:14:10 2020

@author: NINMOYpc
"""
import pickle 
import csv 
from sklearn.preprocessing import StandardScaler


dbfile = open('examplePickle', 'rb')	 
classifier = pickle.load(dbfile)
sc = StandardScaler()
# opening the CSV file 
with open('test.csv', mode ='r')as file: 	
    # reading the CSV file 
    csvFile = csv.reader(file) 

# displaying the contents of the CSV file 
    for lines in csvFile: 
        #print(lines)
        print(classifier.predict(sc.transform([lines])))