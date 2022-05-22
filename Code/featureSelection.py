# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 19:55:11 2021

@author: NINMOYpc
"""
import pandas as pd

#X = [[0, 0], [10, 10],[20,30],[30,30],[40, 30], [80,60], [80,50]]
#y = [0, 0, 1, 0, 1, 1, 1]
dataset = pd.read_csv('MI_Graph.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# import f_classif
from sklearn.feature_selection import f_classif
from sklearn.preprocessing import LabelEncoder, LabelBinarizer
#feature_names = dataset.feature_names
df = pd.read_csv('MI_Graph.csv', index_col=0)
lb = LabelEncoder()
labels = lb.fit_transform((df.index.values))
features = df.values
feature_names = list(df.columns)
print(feature_names)
# create f_classif object
f_value = f_classif(X, y)
# print the name and F-value of each feature
for feature in zip(feature_names, f_value[0]):
    print(feature)

# Create a bar chart
import matplotlib.pyplot as plt
plt.figure(figsize=(100,100))
plt.bar(x=feature_names,height=f_value[0], color="red")
plt.xticks(rotation="vertical")
plt.ylabel("F-value")
plt.title("F-value Comparison")
plt.show()


# import mutual_info_classif
from sklearn.feature_selection import mutual_info_classif
# create mutual_info_classif object
MI_score = mutual_info_classif(X, y, random_state=0)
# Print the name and mutual information score of each feature
for feature in zip(feature_names, MI_score):
    print(feature)
    
# create a bar chart 
plt.figure(figsize=(100,100))
plt.bar(x=feature_names, height=MI_score, color='red')
plt.xticks(rotation='vertical')
plt.ylabel('Mutual Information Score')
plt.title('Mutual Information Score Comparison')
plt.show()

# import SelectKBest
from sklearn.feature_selection import SelectKBest
# create a SelectKBest object
skb = SelectKBest(score_func=f_classif, k=2) 
# set f_classif as ANOVA F-value
# Select top 2 features based on the criteria
# train and transform
X_data_new = skb.fit_transform(X, y)
print("the name of the selected features")
for feature_list_index in skb.get_support(indices=True):
    print(feature_names[feature_list_index])