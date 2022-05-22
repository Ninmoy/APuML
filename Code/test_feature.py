# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 20:28:54 2020

@author: NINMOYpc
"""

import pandas as pd
from math import sqrt
from math import pi
from math import exp
import numpy as np
import requests 
from svm_probability import *
from bs4 import BeautifulSoup
import urllib,bs4


features=list()

f2=open('maliciusttest.txt','r')
a=1
print("for malicious site    :")
urls=f2.readlines()
r2=set()
for URL in urls:
    URL=URL.strip()
    r2.add(URL)

for URL in r2:
    print(URL)
    try : 
         r = requests.get(URL, timeout=120)
         print("request done")
    except :
        print("Can not able to connect to internet ")
        continue
    print(a)
    a=a+1
    temp_list=list()
    temp_list1=list()
    soup = BeautifulSoup(r.content, 'html5lib') 
    print("soup done")
    #print(soup)
    p_js=0
    c_js=0
    for row in soup.findAll('script'): 
        c_js=c_js+1
    if c_js != 0 :
        p_js=1
    print('script done')
    p_ns=0
    c_ns=0
    for row in soup.findAll('noscript'): 
        c_ns=c_ns+1
    if c_ns != 0 :
        p_ns=1
    print('nonscript done')
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
    print('src done')
    p_img=0
    c_img=0
    for row in soup.findAll('img'):
        c_img=c_img+1
    if c_img != 0 :
        p_img=1
    print('img done')
    p_iframe=0
    c_iframe=0
    for row in soup.findAll('iframe'):
        c_iframe=c_iframe+1
    if c_iframe != 0 :
        p_iframe=1
    print('iframe done')
    p_rdirect=0
    c_rdirect=0
    for row in soup.findAll('meta', attrs={'http-equiv':'Refresh' }):
        c_rdirect=c_rdirect+1
    if c_rdirect != 0 :
        p_rdirect=1
    print('refresh done')
    p_lnks=0
    temp=list()
    for row in soup.findAll('a'):
        temp.append(row.get('href'))
    print('href done')
    c_lnks=len(temp)
    if c_lnks != 0 :
        p_lnks=1
    temp_list.append(p_js)
    temp_list.append(p_ns)
    temp_list.append(p_ejs)
    temp_list.append(p_ijs)
    temp_list.append(p_img)
    temp_list.append(p_iframe)
    temp_list.append(p_rdirect)
    temp_list.append(p_lnks)
    temp_list.append(c_js)
    temp_list.append(c_ns)
    temp_list.append(c_ejs)
    temp_list.append(c_ijs)
    temp_list.append(c_img)
    temp_list.append(c_iframe)
    temp_list.append(c_rdirect)
    temp_list.append(c_lnks)
    temp_list1.append(c_js)
    temp_list1.append(c_ns)
    temp_list1.append(c_ejs)
    temp_list1.append(c_ijs)
    temp_list1.append(c_img)
    temp_list1.append(c_iframe)
    temp_list1.append(c_rdirect)
    temp_list1.append(c_lnks)
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
    temp_list1.append(c_sms)
    temp_list1.append(c_tel)
    temp_list1.append(c_apk)
    temp_list1.append(c_mms)
    temp_list.append(c_sms)
    temp_list.append(c_tel)
    temp_list.append(c_apk)
    temp_list.append(c_mms)
    print("sms,tel,apk,mms done")
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
    temp_list1.append(num_fslash)
    temp_list1.append(num_qm)
    temp_list1.append(num_dots)
    temp_list1.append(num_hypen)
    temp_list1.append(num_uscore)
    temp_list1.append(num_eqls)
    temp_list1.append(num_amp)
    temp_list1.append(num_smcolon)
    temp_list1.append(num_digi)
    temp_list.append(num_fslash)
    temp_list.append(num_qm)
    temp_list.append(num_dots)
    temp_list.append(num_hypen)
    temp_list.append(num_uscore)
    temp_list.append(num_eqls)
    temp_list.append(num_amp)
    temp_list.append(num_smcolon)
    temp_list.append(num_digi)
    print('digit, dot, eual etc done')
    try:        traff_rnk=int(bs4.BeautifulSoup(urllib.request.urlopen("http://data.alexa.com/data?cli=10&dat=s&url=" + URL).read(), "xml").find("REACH")["RANK"])
    except:
        traff_rnk=0
    temp_list.append(traff_rnk)
    try:     cntry_traff_rnk=int(bs4.BeautifulSoup(urllib.request.urlopen("http://data.alexa.com/data?cli=10&dat=s&url=" + URL).read(), "xml").find("COUNTRY")["RANK"])
    except:
        cntry_traff_rnk=0
    temp_list1.append(cntry_traff_rnk)
    print("rank done")
    c_f=0
    p_f=0
    for row in soup.findAll('form'): 
        c_f=c_f+1
    if c_f != 0 :
        p_f=1
    temp_list.append(p_f)
    temp_list1.append(c_f)
    temp_list.append(c_f)
    print("form done")
    #prob = y_pred.predict_proba([temp_list1])
    #prob1=prob[0]
    #temp_list.append(prob1[0])
    #temp_list.append(prob1[1])
    #0 is added at lst to reprent it is non malicious/ 1 for malicious
    temp_list.append(0)
    features.append(temp_list)
print("Done for Non malicious site")

fields = ['Presence of JS' ,'Presence of NS' ,'Presence of external JS' ,'Presence of internal JS' ,'Presence of image' ,'Presence of iframe','Presence of redirects' ,'Presence of links' ,'Number of JS' ,'Number of NS' ,'Number of external JS' ,'Number of internal JS' ,'Number of image' ,'Nuber of iframes' ,'Number of redirects' ,'Number of links' ,'Number of sms API call','Number of tel API call','Number of apk API call','Number of mms API call','Length of URL','Number of forward slash','Number of question marks', 'Number of dots', 'Number of hypens' , 'Number of underscore', 'Number of equal signs', 'Number of ampersand' , 'Number of semi-colon','Number of digits','World Traffic rank','Country Traffic rank','presence of form','No of form','probability0','probability1','Output']
#df=pd.DataFrame(features, columns = ['Presence of JS' ,'Presence of NS' ,'Presence of external JS' ,'Presence of internal JS' ,'Presence of image' ,'Presence of iframe','Presence of redirects' ,'Presence of links' ,'Number of JS' ,'Number of NS' ,'Number of external JS' ,'Number of internal JS' ,'Number of image' ,'Nuber of iframes' ,'Number of redirects' ,'Number of links' ,'Number of sms API call','Number of tel API call','Number of apk API call','Number of mms API call','Length of URL','Number of forward slash','Number of question marks', 'Number of dots', 'Number of hypens' , 'Number of underscore', 'Number of equal signs', 'Number of ampersand' , 'Number of semi-colon','Number of digits','World Traffic rank','Country Traffic rank','Output'])

import csv

filename = "NinmoyFinal.csv"
	
# writing to csv file 
with open(filename, 'a') as csvfile: 
	# creating a csv writer object 
	csvwriter = csv.writer(csvfile) 
		
	# writing the fields 
	csvwriter.writerow(fields) 
		
	# writing the data rows 
	csvwriter.writerows(features)