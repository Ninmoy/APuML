# -*- coding: utf-8 -*-
"""
Created on Sat Jan  9 16:16:54 2021

@author: NINMOYpc
"""

import pandas as pd

#X = [[0, 0], [10, 10],[20,30],[30,30],[40, 30], [80,60], [80,50]]
#y = [0, 0, 1, 0, 1, 1, 1]
dataset = pd.read_csv('NBNinmoyFinal.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

#print(y[1])
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)



from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
y_pred = gnb.fit(X_train, y_train)
gnb.probability=True
#gnb.fit(X, y)

import pickle
dbfile = open('NBPickle.pkl', 'wb') 
	
# source, destination 
pickle.dump(y_pred, dbfile)					 
dbfile.close()

#prob = gnb.predict_proba([[1,0,0,1,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,27,4,0,3,0,0,0,0,0,0,0,0]])
#res = y_pred.predict([[1,0,0,1,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,27,4,0,3,0,0,0,0,0,0,0,0]])
#prob1=prob[0]
#print(prob1)
#print(res)


from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, y_pred.predict(X_test)))


#from sklearn import svm
#clf = svm.SVC() 
#clf.probability=True
#clf.fit(X, y)
#for a in X:
    #prob = clf.predict_proba([a])
    #prob1=prob[0]
    #print(prob)