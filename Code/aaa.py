# -*- coding: utf-8 -*-
"""
Created on Sun Nov 29 21:02:08 2020

@author: NINMOYpc
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Sep 19 19:09:52 2020

@author: NINMOYpc
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

dataset = pd.read_csv('for_traning.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
a=classifier.fit(X_train, y_train)


#for saving traning model using pickle model
import pickle
dbfile = open('examplePickle.pkl', 'ab') 
	
# source, destination 
pickle.dump(a, dbfile)					 
dbfile.close()

#import csv 

# opening the CSV file 
#with open('for_prediction_bad.csv', mode ='r')as file: 	
    # reading the CSV file 
    #csvFile = csv.reader(file) 

# displaying the contents of the CSV file 
    #for lines in csvFile: 
        #print(lines)
        #print(classifier.predict(sc.transform([lines])))