# -*- coding: utf-8 -*-
"""
Created on Sat May 16 16:39:40 2020

@author: djhurani
"""

import requests 
from bs4 import BeautifulSoup
import urllib,bs4
import urllib.request
import pickle
#from logistic_regrassion import *
#from svm_probability import *

#URL="https://www.youtube.com"

def retrive_pred(URL):
    try : 
        r = requests.get(URL, timeout=60)
    except :
        return "Error"
    temp_list=list()
    newtemp_list=list()
    soup = BeautifulSoup(r.content, 'html.parser') 
    #print(soup)
    p_js=0
    c_js=0
    for row in soup.findAll('script'): 
        c_js=c_js+1
    if c_js != 0 :
        p_js=1
    print(f"Script present {p_js},count {c_js}")
    
    p_ns=0
    c_ns=0
    for row in soup.findAll('noscript'): 
        c_ns=c_ns+1
    if c_ns != 0 :
        p_ns=1
    print(f"nonScript present {p_ns},count {c_ns}")
    
    p_ejs=0
    p_ijs=0
    c_ejs=0
    for row in soup.findAll('script',attrs='src'): 
        c_ejs=c_ejs+1
    c_ijs=c_js-c_ejs
    if c_ejs != 0 :
        p_ejs=1
    if c_ijs != 0 :
        p_ijs=1
    print(f"Script scr present {p_ejs}, {p_ijs}, count {c_ejs}")
    
    p_img=0
    c_img=0
    for row in soup.findAll('img'):
        c_img=c_img+1
    if c_img != 0 :
        p_img=1
    print(f"Image present {p_img},count {c_img}")
    
    p_iframe=0
    c_iframe=0
    for row in soup.findAll('iframe'):
        c_iframe=c_iframe+1
    if c_iframe != 0 :
        p_iframe=1
    print(f"Iframe present {p_iframe},count {c_iframe}")
    
    p_rdirect=0
    c_rdirect=0
    for row in soup.findAll('meta', attrs={'http-equiv':'Refresh' }):
        c_rdirect=c_rdirect+1
    if c_rdirect != 0 :
        p_rdirect=1
    print(f"redirect present {p_rdirect},count {c_rdirect}")

    p_lnks=0
    temp=list()
    for row in soup.findAll('a'):
        temp.append(row.get('href'))
    c_lnks=len(temp)
    if c_lnks != 0 :
        p_lnks=1
    print(f"href present {p_lnks},count {len(temp)}")
    
    #temp_list.append(p_js)
    #temp_list.append(p_ns)
    #temp_list.append(p_ejs)
    #temp_list.append(p_ijs)
    #temp_list.append(p_img)
    #temp_list.append(p_iframe)
    #temp_list.append(p_rdirect)
    #temp_list.append(p_lnks)
    
    newtemp_list.append(p_js)
    newtemp_list.append(p_ns)
    #newtemp_list.append(p_ejs)
    newtemp_list.append(p_ijs)
    newtemp_list.append(p_img)
    newtemp_list.append(p_iframe)
    #newtemp_list.append(p_rdirect)
    newtemp_list.append(p_lnks)
    
    
    temp_list.append(c_js)
    temp_list.append(c_ns)
    temp_list.append(c_ejs)
    temp_list.append(c_ijs)
    temp_list.append(c_img)
    temp_list.append(c_iframe)
    temp_list.append(c_rdirect)
    temp_list.append(c_lnks)
    c_sms=0
    c_tel=0
    c_apk=0
    c_mms=0
    for s in temp:
        if(isinstance(s,str)):
            if(s.startswith('sms')):
                c_sms=c_sms+1
            if(s.startswith('tel')):
                c_tel=c_tel+1
            if(s.endswith('apk')):
                c_apk=c_apk+1
            if(s.startswith('mms')):
                c_mms=c_mms+1
                
                
    print(f"No of sms: {c_sms}")
    print(f"No of tel: {c_tel}")
    print(f"No of apk: {c_apk}")
    print(f"No of mms: {c_mms}")
    
    
    temp_list.append(c_sms)
    temp_list.append(c_tel)
    temp_list.append(c_apk)
    #temp_list.append(c_mms)
    url_len= len(URL)
    #print(url_len)
    temp_list.append(url_len)
    num_fslash=0
    num_qm=0
    num_dots=0
    num_hypen=0
    num_uscore=0
    num_eqls=0
    num_amp=0
    num_smcolon=0
    num_digi=0
    for i in URL:
        if i== '/' :
            num_fslash=num_fslash+1
        if i=='?' :
            num_qm=num_qm+1
        if i=='.' :
            num_dots=num_dots+1
        if i=='-' :
            num_hypen=num_hypen+1
        if i=='_' :
            num_uscore=num_uscore+1
        if i=='=' :
            num_eqls=num_eqls+1
        if i=='&' :
            num_amp=num_amp+1
        if i==';' :
            num_smcolon=num_smcolon+1
        if i=='0' or i=='1' or i=='2' or i=='3' or i=='4' or i=='5' or i=='6' or i=='7' or i=='8' or i=='9' :
            num_digi=num_digi+1
    temp_list.append(num_fslash)
    temp_list.append(num_qm)
    temp_list.append(num_dots)
    temp_list.append(num_hypen)
    temp_list.append(num_uscore)
    temp_list.append(num_eqls)
    temp_list.append(num_amp)
    #temp_list.append(num_smcolon)
    temp_list.append(num_digi)
    
    print(f"No of fslash: {num_fslash}")
    print(f"No of qm: {num_qm}")
    print(f"No of dots: {num_dots}")
    print(f"No of hypen: {num_hypen}")
    print(f"No of uscore: {num_uscore}")
    print(f"No of eqls: {num_eqls}")
    print(f"No of amp: {num_amp}")
    print(f"No of smcolon: {num_smcolon}")
    print(f"No of digi: {num_digi}")
    
    
    try:
        traff_rnk=int(bs4.BeautifulSoup(urllib.request.urlopen("http://data.alexa.com/data?cli=10&dat=s&url=" + URL).read(), "xml").find("REACH")["RANK"])
    except Exception as err:
        print(err)
        traff_rnk=0
        newtraff_rnk=0
    if traff_rnk>0:
        newtraff_rnk=2
    temp_list.append(traff_rnk)
    newtemp_list.append(newtraff_rnk)
    
    try:
        cntry_traff_rnk=int(bs4.BeautifulSoup(urllib.request.urlopen("http://data.alexa.com/data?cli=10&dat=s&url=" + URL).read(), "xml").find("COUNTRY")["RANK"])
    except Exception as err:
        print(err)
        cntry_traff_rnk=0
        newcntry_traff_rnk=0
    if cntry_traff_rnk>0:
        newcntry_traff_rnk=2
    temp_list.append(cntry_traff_rnk)
    newtemp_list.append(newcntry_traff_rnk)
    print("Rank done")
    c_f=0
    p_f=0
    for row in soup.findAll('form'): 
        c_f=c_f+1
    if c_f != 0 :
        p_f=1
    #temp_list.append(p_f)
    newtemp_list.append(p_f)
    temp_list.append(c_f)
    print("form done")
    
    dbfileN = open('NBPickle.pkl', 'rb')	 
    y_pred = pickle.load(dbfileN)
    prob = y_pred.predict_proba([temp_list])
    prob1=prob[0]
    if prob1[0]>0.5:
        newtemp_list.append(2)
        newtemp_list.append(0)
    else:
        newtemp_list.append(0)
        newtemp_list.append(2)
    
    #temp_list.append(0)
   # temp_list.append(0)
    print(newtemp_list)
    #import csv 

    dbfile = open('examplePickle.pkl', 'rb')	 
    classifier = pickle.load(dbfile)

    r=classifier.predict([newtemp_list])
    print("Value of r : ", r)
    if r[0]==0:
        return "Non malicious"
    else:
        return "Malicious"

#retrive_pred("https://m.facebook.com/notifications.php")
import time        
data= "https://gmail.com"
startTime=time.time()
output = retrive_pred(data)
executionTime=(time.time()-startTime)
print("Excution Time : ",executionTime)
print(output)
