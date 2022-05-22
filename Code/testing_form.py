# -*- coding: utf-8 -*-
"""
Created on Sat Jan  2 19:35:45 2021

@author: NINMOYpc
"""

import pandas as pd
from math import sqrt
from math import pi
from math import exp
import numpy as np
import requests 
from bs4 import BeautifulSoup
import urllib,bs4


features=list()


def test(URL):
    print(URL)
    try : 
        r = requests.get(URL)
        print("request done")
    except :
        print("Can not able to connect to internet ")
        return
    soup = BeautifulSoup(r.content, 'html5lib') 
    print("soup done")
    #print(soup)
    p_js=0
    c_js=0
    for row in soup.findAll('form'): 
        c_js=c_js+1
    if c_js != 0 :
        p_js=1
    print('from done no of from: ')
    print(c_js)
test("https://millanobet263.com")