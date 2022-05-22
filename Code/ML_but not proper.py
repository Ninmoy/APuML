import pandas as pd
from math import sqrt
from math import pi
from math import exp
import numpy as np
import requests 
from bs4 import BeautifulSoup
import urllib,bs4
#algorithm to extract mobile specific urls
y=set()
r1=set()
f1=open('feed.txt','r') #feed.txt contains malicious sites from OpenPhish
y=f1.readlines()
for z in y:
    res=False
    url=z.strip()
    x1=url.split('/')
    x2=list()
    for i in x1:
        x3=i.split('.')
        for j in x3:
            x2.append(j)
    for i in x2:
        if (i=='mobi' or i=='m' or i=='mobile' or i=="touch" or i=='3g' or i=='sp' or i=='s' or i=='mini' or i=='mobileweb' or i=='t' or i=='?m=1'or i=='mobil' or i=='m_home' ):
            res=True
    if(res==True):
        r1.add(url)
num_mal=len(r1)
print("Number of mobile specific malicious sites : ", num_mal)
features=list()
print("for malicious site") 
#extractig features from malicious sites 
for URL in r1:
    try : 
        r = requests.get(URL)
    except :
        continue
    print(URL)
    temp_list=list()
    soup = BeautifulSoup(r.content, 'html5lib') 
    #print(soup)
    p_js=0
    c_js=0
    for row in soup.findAll('script'): 
        c_js=c_js+1
    if c_js != 0 :
        p_js=1 
    p_ns=0
    c_ns=0
    for row in soup.findAll('noscript'): 
        c_ns=c_ns+1
    if c_ns != 0 :
        p_ns=1
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
    p_img=0
    c_img=0
    for row in soup.findAll('img'):
        c_img=c_img+1
    if c_img != 0 :
        p_img=1
    p_iframe=0
    c_iframe=0
    for row in soup.findAll('iframe'):
        c_iframe=c_iframe+1
    if c_iframe != 0 :
        p_iframe=1
    p_rdirect=0
    c_rdirect=0
    for row in soup.findAll('meta', attrs={'http-equiv':'Refresh' }):
        c_rdirect=c_rdirect+1
    if c_rdirect != 0 :
        p_rdirect=1
    
    p_lnks=0
    temp=list()
    for row in soup.findAll('a'):
        temp.append(row.get('href'))
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
    temp_list.append(c_sms)
    temp_list.append(c_tel)
    temp_list.append(c_apk)
    temp_list.append(c_mms)
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
    temp_list.append(num_smcolon)
    temp_list.append(num_digi)
    
    
    
    try:  traff_rnk=int(bs4.BeautifulSoup(urllib.request.urlopen("http://data.alexa.com/data?cli=10&dat=s&url=" + URL).read(), "xml").find("REACH")["RANK"])
    except:
        traff_rnk=0
    temp_list.append(traff_rnk)
    try:
        cntry_traff_rnk=int(bs4.BeautifulSoup(urllib.request.urlopen("http://data.alexa.com/data?cli=10&dat=s&url=" + URL).read(), "xml").find("COUNTRY")["RANK"])
    except:
        cntry_traff_rnk=0
    temp_list.append(cntry_traff_rnk)
    temp_list.append(1)
    features.append(temp_list)
#extracting features from legitimate sites
print("Done for malicious site")
f2=open('legitimate_sites.txt','r')
print("for legitimate site    :")
urls=f2.readlines()
r2=set()
for URL in urls:
    URL=URL.strip()
    r2.add(URL)
for URL in r2:
    try : 
        r = requests.get(URL)
    except :
        continue
    print(URL)
    temp_list=list()
    soup = BeautifulSoup(r.content, 'html5lib') 
    #print(soup)
    p_js=0
    c_js=0
    for row in soup.findAll('script'): 
        c_js=c_js+1
    if c_js != 0 :
        p_js=1
    p_ns=0
    c_ns=0
    for row in soup.findAll('noscript'): 
        c_ns=c_ns+1
    if c_ns != 0 :
        p_ns=1 
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
    p_img=0
    c_img=0
    for row in soup.findAll('img'):
        c_img=c_img+1
    if c_img != 0 :
        p_img=1
    p_iframe=0
    c_iframe=0
    for row in soup.findAll('iframe'):
        c_iframe=c_iframe+1
    if c_iframe != 0 :
        p_iframe=1
    p_rdirect=0
    c_rdirect=0
    for row in soup.findAll('meta', attrs={'http-equiv':'Refresh' }):
        c_rdirect=c_rdirect+1
    if c_rdirect != 0 :
        p_rdirect=1
    p_lnks=0
    temp=list()
    for row in soup.findAll('a'):
        temp.append(row.get('href'))
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
    temp_list.append(c_sms)
    temp_list.append(c_tel)
    temp_list.append(c_apk)
    temp_list.append(c_mms)
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
    temp_list.append(num_smcolon)
    temp_list.append(num_digi) 
    try:        traff_rnk=int(bs4.BeautifulSoup(urllib.request.urlopen("http://data.alexa.com/data?cli=10&dat=s&url=" + URL).read(), "xml").find("REACH")["RANK"])
    except:
        traff_rnk=0
    temp_list.append(traff_rnk)
    try:     cntry_traff_rnk=int(bs4.BeautifulSoup(urllib.request.urlopen("http://data.alexa.com/data?cli=10&dat=s&url=" + URL).read(), "xml").find("COUNTRY")["RANK"])
    except:
        cntry_traff_rnk=0
    temp_list.append(cntry_traff_rnk)
    temp_list.append(0)
    features.append(temp_list)
print("Done for lagitimate site")
df=pd.DataFrame(features, columns = ['Presence of JS' ,'Presence of NS' ,'Presence of external JS' ,'Presence of internal JS' ,'Presence of image' ,'Presence of iframe','Presence of redirects' ,'Presence of links' ,'Number of JS' ,'Number of NS' ,'Number of external JS' ,'Number of internal JS' ,'Number of image' ,'Nuber of iframes' ,'Number of redirects' ,'Number of links' ,'Number of sms API call','Number of tel API call','Number of apk API call','Number of mms API call','Length of URL','Number of forward slash','Number of question marks', 'Number of dots', 'Number of hypens' , 'Number of underscore', 'Number of equal signs', 'Number of ampersand' , 'Number of semi-colon','Number of digits','World Traffic rank','Country Traffic rank','Output'])
print("dfffffffffff  : ", df)

file = open('read.txt', 'w') 
file.write(df) 
file.close() 
#Implemeting Naive Bayes from scratch to add as a feature
# Splitting the dataset based on class value
def split(dset):
	split = dict()
	for i in range(len(dset)):
		feature_vector = dset[i]
		val_cls = feature_vector[-1]
		if (val_cls not in split):
			split[val_cls] = list()
		split[val_cls].append(feature_vector)
	return split
def nums_mean(nums):
	return sum(nums)/float(len(nums))
def nums_stdev(nums):
	nums_avg = nums_mean(nums)
	vari = sum([(x-nums_avg)**2 for x in nums]) / float(len(nums)-1)
	return sqrt(vari)
def smrze(dset):
	smrize = [(nums_mean(column), nums_stdev(column), len(column)) for column in zip(*dset)]
	del(smrize[-1])
	return smrize
def summarize_by_class(dset):
	separated = split(dset)
	smries = dict()
	for val_cls, rows in separated.items():
		smries[val_cls] = smrze(rows)
	return smries
def cal_prob(x, mn, std):
    expo = exp(-((x-mn)**2 / (2 * std**2 )))
    try:
        return (1 / (sqrt(2 * pi) * std)) * expo
    except:
        return 1
def cal_cls_prob(smries, row):
	tr = sum([smries[label][0][2] for label in smries])
	probs = dict()
	for cls_val, cls_smries in smries.items():
		probs[cls_val] = smries[cls_val][0][2]/float(tr)
		for i in range(len(cls_smries)):
			mn, std, _ = cls_smries[i]
			probs[cls_val] *= cal_prob(row[i], mn, std)
	return probs
arr=df.to_numpy()
#Calculating class probabilities
temp0=list()
temp1=list()
smries = summarize_by_class(arr)
for i in range(len(df)) :
    probability = cal_cls_prob(smries, arr[i])
    print("probability  : ",probability)
    temp0.append(probability[0])
    temp1.append(probability[1])
#inserting the calculated class probability 
da_frm=pd.DataFrame(arr)    
da_frm.insert(32, 'benign_probability', temp0)  
da_frm.insert(33, 'malicious_probability', temp1)
da_frm.columns= ['Presence of JS' ,'Presence of NS' ,'Presence of external JS' ,'Presence of internal JS' ,'Presence of image' ,'Presence of iframe','Presence of redirects' ,'Presence of links' ,'Number of JS' ,'Number of NS' ,'Number of external JS' ,'Number of internal JS' ,'Number of image' ,'Number of iframes' ,'Number of redirects' ,'Number of links' ,'Number of sms API call','Number of tel API call','Number of apk API call','Number of mms API call','Length of URL','Number of forward slash','Number of question marks', 'Number of dots', 'Number of hypens' , 'Number of underscore', 'Number of equal signs', 'Number of ampersand' , 'Number of semi-colon','Number of digits','World Traffic rank','Country Traffic rank','benign_probability','malicious_probability','Output']
#Feature analysis process
am = da_frm.iloc[:,:-1] 
ar = da_frm.iloc[:,-1]
from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt
mdl = ExtraTreesClassifier()
mdl.fit(am,ar)
features_imp = pd.Series(mdl.feature_importances_, index=am.columns)
features_imp.nlargest(21).plot(kind='barh')
plt.show()
#dropping insignificant features
final_df=df.drop(['malicious_probability','benign_probability','Presence of external JS' ,'Number of external JS','Number of mms API call', 'Number of semi-colon','Number of redirects','Number of sms API call','Presence of redirects','Number of apk API call','Number of tel API call','Number of underscore','Number of ampersand'],axis=1)
final_df.columns
#Final feature matrix and response vector
final_X=final_df.iloc[:,:-1].values
final_y=final_df.iloc[:,-1].values
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(final_X,final_y, test_size = 0.1, random_state=0)
#Applying various Classification Algorithms
from sklearn.linear_model import LogisticRegression
classifierLR = LogisticRegression(random_state = 0)
classifierLR.fit(X_train,y_train)
y_predLR=classifierLR.predict(X_test)
from sklearn.metrics import confusion_matrix
cmLR = confusion_matrix(y_test,y_predLR)
from sklearn import metrics
accuracyLR=metrics.accuracy_score(y_test,y_predLR)
precisionLR=cmLR[0][0]/(cmLR[0][0]+cmLR[1][0])
recallLR=cmLR[0][0]/(cmLR[0][0]+cmLR[0][1])
f1_scoreLR=(2*precisionLR*recallLR)/(precisionLR+recallLR)
TPLR = cmLR[0][0]
FPLR = cmLR[0][1]
FNLR = cmLR[1][0]
TNLR = cmLR[1][1]
TPRLR=TPLR/(TPLR+FNLR)
TNRLR=TNLR/(TNLR+FPLR)
print("TPR for Logistic Regression algorithm : ",TPRLR)
print("TNR for Logistic Regression algorithm : ",TNRLR)
print("Accuracy of Logistic Regression algorithm : ",accuracyLR)
print("Precision for Logistic Regression algorithm : ",precisionLR) 
print("f1 score for Logistic Regression algorithm : ",f1_scoreLR)
from sklearn.neighbors import KNeighborsClassifier
classifierKNN = KNeighborsClassifier(n_neighbors = 9, metric = 'minkowski' , p=2)
classifierKNN.fit(X_train,y_train)
y_predKNN=classifierKNN.predict(X_test)
cmKNN = confusion_matrix(y_test,y_predKNN)
accuracyKNN=metrics.accuracy_score(y_test,y_predKNN)
precisionKNN=cmKNN[0][0]/(cmKNN[0][0]+cmKNN[1][0])
recallKNN=cmKNN[0][0]/(cmKNN[0][0]+cmKNN[0][1])
f1_scoreKNN=(2*precisionKNN*recallKNN)/(precisionKNN+recallKNN)
TPKNN = cmKNN[0][0]
FPKNN = cmKNN[0][1]
FNKNN = cmKNN[1][0]
TNKNN = cmKNN[1][1]
TPRKNN=TPKNN/(TPKNN+FNKNN)
TNRKNN=TNKNN/(TNKNN+FPKNN)
print("TPR for K Nearest Neighbors algorithm : ",TPRKNN)
print("TNR for K Nearest Neighbors algorithm : ",TNRKNN)
print("Accuracy of K Nearest Neighbors algorithm : ",accuracyKNN)
print("Precision for K Nearest Neighbor algorithm : ",precisionKNN)
print("f1 score for K Nearest Neighbor algorithm : ",f1_scoreKNN) 
from sklearn.svm import SVC
classifierSVM = SVC(kernel='rbf',random_state=0)
classifierSVM.fit(X_train,y_train)
y_predSVM=classifierSVM.predict(X_test)
#Calculating the accuracy, precision and recall for Support Vector Machine by checking it against the test set
cmSVM = confusion_matrix(y_test,y_predSVM)
accuracySVM=metrics.accuracy_score(y_test,y_predSVM)
precisionSVM=cmSVM[0][0]/(cmSVM[0][0]+cmSVM[1][0])
recallSVM=cmSVM[0][0]/(cmSVM[0][0]+cmSVM[0][1])
f1_scoreSVM=(2*precisionSVM*recallSVM)/(precisionSVM+recallSVM)
TPSVM = cmSVM[0][0]
FPSVM = cmSVM[0][1]
FNSVM = cmSVM[1][0]
TNSVM = cmSVM[1][1]
TPRSVM=TPSVM/(TPSVM+FNSVM)
TNRSVM=TNSVM/(TNSVM+FPSVM)
print("TPR for Support Vector Machine algorithm : ",TPRSVM)
print("TNR for Support Vector Machine algorithm : ",TNRSVM)
print("Accuracy of Support Vector Machine algorithm : ",accuracySVM)
print("Precision for Support Vector Machine algorithm : ",precisionSVM)
print("f1 score for Support Vector Machine algorithm : ",f1_scoreSVM) 
from sklearn.tree import DecisionTreeClassifier
classifierDT = DecisionTreeClassifier(criterion='entropy',random_state=0)
classifierDT.fit(X_train,y_train)
y_predDT=classifierDT.predict(X_test)
cmDT = confusion_matrix(y_test,y_predDT)
accuracyDT=metrics.accuracy_score(y_test,y_predDT)
precisionDT=cmDT[0][0]/(cmDT[0][0]+cmDT[1][0])
recallDT=cmDT[0][0]/(cmDT[0][0]+cmDT[0][1])
f1_scoreDT=(2*precisionDT*recallDT)/(precisionDT+recallDT)
TPDT = cmDT[0][0]
FPDT = cmDT[0][1]
FNDT = cmDT[1][0]
TNDT = cmDT[1][1]
TPRDT=TPDT/(TPDT+FNDT)
TNRDT=TNDT/(TNDT+FPDT)
print("TPR for Decision Tree algorithm : ",TPRDT)
print("TNR for Decision Tree algorithm : ",TNRDT)
print("Accuracy of Decision Tree algorithm : ",accuracyDT)
print("Precision for Decision Tree algorithm: ",precisionDT) 
print("f1 score for Decision Tree algorithm: ",f1_scoreDT)
from sklearn.ensemble import RandomForestClassifier
classifierRFC = RandomForestClassifier(n_estimators = 15,criterion='entropy',random_state=0)
classifierRFC.fit(X_train,y_train)
y_predRFC=classifierRFC.predict(X_test)
cmRFC = confusion_matrix(y_test,y_predRFC)
accuracyRFC=metrics.accuracy_score(y_test,y_predRFC)
precisionRFC=cmRFC[0][0]/(cmRFC[0][0]+cmRFC[1][0])
recallRFC=cmRFC[0][0]/(cmRFC[0][0]+cmRFC[0][1])
f1_scoreRFC=(2*precisionRFC*recallRFC)/(precisionRFC+recallRFC)
TPRFC = cmRFC[0][0]
FPRFC = cmRFC[0][1]
FNRFC = cmRFC[1][0]
TNRFC = cmRFC[1][1]
TPRRFC=TPRFC/(TPRFC+FNRFC)
TNRRFC=TNRFC/(TNRFC+FPRFC)
print("TPR for Random Forest Classification algorithm : ",TPRRFC)
print("TNR for Random Forest Classification algorithm : ",TNRRFC)
print("Accuracy of Random Forest Classification algorithm : ",accuracyRFC)
print("Precision for Random Forest Classification algorithm : ",precisionRFC)
print("f1 score for Random Forest Classification algorithm : ",f1_scoreRFC) 
#saving the RFC model
print("Dumping")
import pickle
pickle.dump(classifierRFC,open('saved_model.pkl','wb'),protocol=2)
print("Dumped")