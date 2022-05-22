# -*- coding: utf-8 -*-
"""
Created on Sat Sep 19 19:09:52 2020

@author: NINMOYpc
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('4testNinmoyFinal.csv')

#dataset = pd.read_csv('test_Mobile_specific .csv')
#dataset = pd.read_csv('test1_JavaScript .csv')
#dataset = pd.read_csv('test1_HTML.csv')
#dataset = pd.read_csv('test_URL.csv')
#dataset = pd.read_csv('test1_Website_specific.csv')
#dataset = pd.read_csv('test1_Fake_form.csv')
#dataset = pd.read_csv('test_Naive_Bayes .csv')

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .3, random_state = 1)


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#logistic regrassion
#from sklearn.linear_model import LogisticRegression
#classifier = LogisticRegression(random_state = 1)
#a=classifier.fit(X_train, y_train)

#Random Forest
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier()
a=classifier.fit(X_train, y_train)

#SVM
#from sklearn import svm
#classifier = svm.SVC(kernel='linear', gamma = 'auto')
#a=classifier.fit(X_train, y_train)

#Dession Tree
#from sklearn import tree
#classifier = tree.DecisionTreeClassifier()
#a=classifier.fit(X_train, y_train)

#neural Network
#from sklearn.neural_network import MLPClassifier
#classifier = MLPClassifier(solver='lbfgs', alpha=1e-1, hidden_layer_sizes=(7, 2), random_state=1)
#a=classifier.fit(X_train, y_train)

#KNeighbors Classificaton
#from sklearn.neighbors import KNeighborsClassifier
#classifier = KNeighborsClassifier(n_neighbors=3)
#a=classifier.fit(X_train, y_train)

#r=classifier.predict([[129,9,0,3,1,0,0,0,17]])
#print("Output",r)
from sklearn.metrics import accuracy_score
result=accuracy_score(classifier.predict(X_test), y_test)
print("Accuracy : ",result)
#for saving traning model using pickle model

#from sklearn.model_selection import cross_val_score
#scores = cross_val_score(classifier, X, y, cv=4)
#print("cross validation :",scores)

#import numpy as np
from sklearn.metrics import confusion_matrix
cnf_matrix = confusion_matrix(y_test, classifier.predict(X_test))
print(cnf_matrix)
#[[1 1 3]
# [3 2 2]
# [1 3 1]]

FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)  
FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
TP = np.diag(cnf_matrix)
TN = cnf_matrix.sum() - (FP + FN + TP)

FP = FP.astype(float)
FN = FN.astype(float)
TP = TP.astype(float)
TN = TN.astype(float)

# Sensitivity, hit rate, recall, or true positive rate
TPR = TP/(TP+FN)
# Specificity or true negative rate
TNR = TN/(TN+FP) 
# Precision or positive predictive value
PPV = TP/(TP+FP)
# Negative predictive value
NPV = TN/(TN+FN)
# Fall out or false positive rate
FPR = FP/(FP+TN)
# False negative rate
FNR = FN/(TP+FN)
# False discovery rate
FDR = FP/(TP+FP)
# Overall accuracy
ACC = (TP+TN)/(TP+FP+FN+TN)

print("TPR : ",TPR)
print("TNR : ",TNR)
print("PPV : ",PPV)
print("ACC : ",ACC)

import pickle
dbfile = open('examplePickle.pkl', 'wb') 
	
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