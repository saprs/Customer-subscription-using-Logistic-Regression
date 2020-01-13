# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 16:59:57 2018

@author: vibeeshm
"""

import numpy as np
import matplotlib . pyplot as plt
import pandas as pd 

dataset = pd .read_csv ('C:/Users/madha/Documents/Sec Sem/Machine Learning/Assignment 2/bank.csv', sep = ';')


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