# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 17:09:00 2018

@author: vibeeshm
"""
 
 
import numpy as np
import pandas as pd



dataset = pd .read_csv ('C:/Users/vibeeshm/Desktop/bank-small-train.csv', sep = ';')


def changingintointegervalues(dataset):
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
     
dataset = changingintointegervalues(dataset)

#Splitting X and Y from dataset
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Adding Bias Term
x = np.c_[np.ones((x.shape[0])),x]

kf = KFold(n_splits=10)
kfold = kf.get_n_splits(dataset)
print(kfold)


nEpoch = 50
alpha = 0.01
          
accscore = np.arange(10.0)
confmatrix = np.arange(10.0)
precision = np.arange(10.0)
recall = np.arange(10.0)
f1 = np.arange(10.0)
i = 0
j = 0

def confusionmatrix(ypred, y_test):
    
    #if(ypred >= 0 and ypred <0.5):
     #   ypred = 0
    #else:
     #   ypred = 1
   
    cm = np.matrix('0 0 ; 0 0')
    for j in np.arange(0, (y_test.shape[0])):
           
           #Setting the thresold value
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
        #print(cm)
           j = j+1
    print(cm)
    return cm
     

def sigmoidFunction(z):
           return 1.0 / (1.0 + np.exp(-z))         
#def accuracy()
#train and test splitting
    
for train_index, test_index in kf.split(dataset):
       
       # print("TRAIN", train_index, "TEST", test_index)
       x_train, x_test = x[train_index], x[test_index]
       y_train, y_test = y[train_index], y[test_index]
       
       #intializing w value
       w = np.random.uniform(size=(x_train.shape[1],))
       
       #Finding Model
       for epoch in np.arange(0,nEpoch):
          hypothesis = sigmoidFunction(x_train.dot(w))
          error = hypothesis - y_train
          loss = np.sum(error**2)
          gradient = x_train.T.dot(error)
          w = w - alpha*gradient
       
       #ypredicaiton
       ypred = sigmoidFunction(x_test.dot(w))
       
       # Creating Conformaiton Matrix
       confmat = confusionmatrix(ypred, y_test)
       
       #Finding Accuracy, Precision, Recall, F1
       accscore[i] = (confmat[0,0] + confmat[1,1])/(confmat.sum())
       precision[i] = (confmat[0,0]/(confmat[0,0]+confmat[1,0]))
       if (confmat[0,0] == 0 and confmat[0,1] == 0):
           recall[i] = (confmat[1,1])/(confmat[1,1]+confmat[0,1])
       else:
           recall[i] = (confmat[0,0])/(confmat[0,0]+confmat[0,1])
       f1[i] = 2.0 * ((precision[i]*recall[i])/(precision[i]+recall[i]))
       i = i+1
       
#average scores       
avg_accscore = np.mean(accscore)       
avg_precision = np.mean(precision)
avg_recall = np.mean(recall)
avg_f1 = np.mean(f1)


    


           
       
