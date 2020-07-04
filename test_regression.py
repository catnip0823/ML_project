#!/usr/bin/env python
# coding: utf-8

# In[1]:


import random
from sklearn import preprocessing
test_size = 80
test_list = random.sample(range(0, 500), test_size)
test_list.sort()


# In[2]:


import functions


# In[3]:


import csv

train_matrix = []
test_matrix = []
train_label = []
test_label = []

with open('dataset/Admission_Predict_data/Admission_Predict_Ver1.1.csv', encoding='utf-8') as f:
    reader = csv.reader(f)
    header = next(reader) # title
    index_test_list = 0
    index_row_list = 0
    for row in reader:
        train_row = []
        for i in range(1, len(row) - 1):
            train_row.append(float(row[i]))
        test_row = float(row[-1])
#         test_row = int(round(test_row * 10))
        
        if index_test_list < test_size and test_list[index_test_list] == index_row_list:
            test_matrix.append(train_row)
            test_label.append(test_row)
            index_test_list += 1
        else:
            train_matrix.append(train_row)
            train_label.append(test_row)
        index_row_list += 1
        


# In[4]:


import numpy as np
x = np.array(train_matrix)
x = preprocessing.scale(x) # normalize
y = np.array(train_label)
x_test = np.array(test_matrix)
x_test = preprocessing.scale(x_test) # normalize
y_test = np.array(test_label)


# In[8]:


from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
import datetime

from sklearn import linear_model
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn import svm
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import RandomForestRegressor


algorithm_name = ['Regression(Lasso)', 'KNN', 'Decision Tree', 'SVM(Linear)', 'AdaBoost', 'Random Forest']
algorithm = []
algorithm.append(linear_model.Lasso(alpha = 0.00114115))
algorithm.append(KNeighborsRegressor(n_neighbors = 31))
algorithm.append(DecisionTreeRegressor(max_depth = 4))
algorithm.append(svm.SVR(kernel = 'linear', C = 0.0918484848484))
algorithm.append(AdaBoostRegressor(n_estimators = 10))
algorithm.append(RandomForestRegressor(n_estimators = 13, max_depth = 4))


for i in range(len(algorithm_name)):
    kfold = KFold(n_splits = 5, shuffle = False)
    index = kfold.split(X = x, y = y)
    for train_index, val_index in index:
        starttime = datetime.datetime.now()
        algorithm[i].fit(x[train_index], y[train_index]) # train
        y_pred = algorithm[i].predict(x[val_index]) # predict
        accuracy1 = r2_score(list(y_pred), list(y[val_index]))
        y_pred = algorithm[i].predict(x_test) # predict
        accuracy = r2_score(list(y_pred), list(y_test))
        endtime = datetime.datetime.now()
        time = (endtime - starttime).microseconds
    print('algorithm:', algorithm_name[i])
    print("val accuracy:", accuracy1)
    print("test accuracy:", accuracy)
    print("time:", time, 'microsecond')
    print()


# In[ ]:




