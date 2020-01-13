# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 17:58:20 2018

@author: vibeeshm
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 17:49:40 2018

@author: vibeeshm
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 00:17:00 2018

@author: vibeeshm
"""

import numpy as np
import matplotlib . pyplot as plt
import pandas as pd 
from sklearn.model_selection import KFold
#from sklearn import preprocessing

train = pd.read_csv("C:/Users/vibeeshm/Desktop/bank-small-train.csv", sep = ";")
test = pd.read_csv("C:/Users/vibeeshm/Desktop/bank-small-test.csv", sep = ";")

#changing the Cat value
def changingintointegervalue(dataset):
       dataset.groupby(['housing']).groups.keys()
       cat_no = {"housing" : {"yes" : 1, "no" : 0}}
       dataset.replace(cat_no, inplace= True)
       
       dataset.groupby(['job']).groups.keys()
       cat_no = {'job' : {"unknown":0, "admin." : 1, "blue-collar":2 ,"entrepreneur" : 3, "housemaid" : 4, "management" : 5, "retired":6, "self-employed" :7, "services":8, "student":9, "technician" : 10, "unemployed" : 11}}
       dataset.replace(cat_no, inplace= True)
       
       dataset.groupby(["marital"]).groups.keys()
       cat_no = {"marital":{"divorced":0,"single":1, "married":2}}
       dataset.replace(cat_no, inplace = True)
       
       dataset.groupby(["loan"]).groups.keys()
       cat_no = {"loan" : {"yes" :1, "no" : 0}}
       dataset.replace(cat_no, inplace = True)
       
       dataset.groupby(["education"]).groups.keys()
       cat_no = {"education":{"primary":1,"secondary":2,"tertiary":3 ,"unknown":0,}}
       dataset.replace(cat_no, inplace = True)
       
       dataset.groupby(['default']).groups.keys()
       cat_no = {"default" : {"yes" : 1, "no" : 0}}
       dataset.replace(cat_no, inplace= True)
       
       dataset.groupby(['contact']).groups.keys()
       cat_no = {"contact" : {"unknown" : 0, "cellular" : 1, "telephone" : 2}}
       dataset.replace(cat_no, inplace= True)
       
       dataset.groupby(['month']).groups.keys()
       cat_no = {"month":{"jan":1,"feb":2,"mar":3,"apr":4,"may":5,"jun":6,"jul":7,"aug":8,"sep":9,"oct":10,"nov":11,"dec":12}}
       dataset.replace(cat_no, inplace = True)
       
       dataset.groupby(["poutcome"]).groups.keys()
       cat_no = {"poutcome":{"unknown":0,"failure":1,"other":2,"success":3}}
       dataset.replace(cat_no, inplace = True)

       dataset.groupby(["y"]).groups.keys()
       cat_no = {"y" : {"yes" : 1, "no" : 0}}
       dataset.replace(cat_no, inplace= True)
       
       return dataset

train = changingintointegervalue(train)
test = changingintointegervalue(test)

x_train = (train.iloc[: , : -1].values).astype(float)
y_train = (train.iloc[:, -1].values).astype(float)
x_test = (test.iloc[: , : -1].values).astype(float)
y_test = (test.iloc[: , -1].values).astype(float)

def standardization(x):
       for i in np.arange(0, ((x.shape[1]))):
              col = x[:, i]
              mean = (x[:,i]).mean()
              sd = (x[:,i]).std()
              col = ((col-mean)/(sd))
              x[:, i] = col
                     #print(dataset[df[0]])
       return x

x_train = standardization(x_train)
x_test = standardization(x_test) 


x_train = np.c_[np.ones((x_train.shape[0])),x_train]
x_test = np.c_[np.ones((x_test.shape[0])),x_test]
 
w = np.random.uniform(size=(x_train.shape[1],))

#changing alpha values
nEpoch = 50
alpha = 0.1

z = x_train.dot(w)

def sigmafunction(z):
       return (1.0/(1.0+np.exp(-z))).astype(float)

for epoch in np.arange(0, nEpoch):
      hypo = sigmafunction(z)
      error = (hypo-y_train)
      gradient = x_train.T.dot(error)
      w = w-alpha*gradient

ypred = sigmafunction(x_test.dot(w))

def confusionmatrix(ypred, y_test):
       cm = np.matrix('0 0 ; 0 0')
       for j in np.arange(0, (y_test.shape[0])):                
               if(ypred[j] < 0.5):
                      ypred[j] = 0.0
               elif(ypred[j] > 0.5):
                      ypred[j] = 1.0
                      
               if ypred[j] == 0 and y_test[j] == 0:
                      cm[1,1] = cm[1,1] + 1
               elif ypred[j] == 1 and y_test[j] == 1:
                      cm[0,0] = cm[0,0]+ 1
               elif y_test[j] ==1 and ypred[j] == 0:
                     cm[0,1] = cm[0,1] + 1
               elif y_test[j] == 0 and ypred[j] == 1: 
                     cm[1,0] = cm[1,0] + 1   
               j = j+1
       return cm

confmat = confusionmatrix(ypred, y_test)
print(confmat)

accuracy = (confmat[0,0] + confmat[1,1])/confmat.sum()
precision =  (confmat[0,1]/(confmat[0,1]+confmat[1,0]))
recall = (confmat[1,1])/(confmat[1,1]+confmat[0,1])
f1 = 2.0 * ((precision * recall) / (precision + recall)) 

