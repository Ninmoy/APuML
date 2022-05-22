# -*- coding: utf-8 -*-
"""
Created on Fri May 21 17:15:26 2021

@author: NINMOYpc
"""

from flask import Flask
app = Flask(__name__)

@app.route('/')
def hello_world():
   return "Hello World"

if __name__ == '__main__':
   app.run()