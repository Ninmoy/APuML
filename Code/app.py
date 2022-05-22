# -*- coding: utf-8 -*-
"""
Created on Sat May 16 04:57:24 2020

@author: djhurani
"""


from flask import Flask, request, redirect, url_for, flash, jsonify
from model import *
import pickle
import json

data="https://jio.com"
output = retrive_pred(data)
print(output)

app = Flask(__name__)

@app.route('/',methods=['GET'])
def make_pred():
    data="https://www.youtube.com"
    pred_value = np.array2string(model.predict(data))
    output = retrive_pred(data)
    print(output)
    # return output
    #return "Home url"

@app.route('/predict',methods=['GET','POST'])
def api_pred():
    data=request.get_json()
    print(data)
    s=data['url']
    output=retrive_pred(s)
    return output
    
    

if __name__ == '__main__':
    model=pickle.load(open('examplePickle.pkl','rb'))
    make_pred()
    app.run(debug=True)
