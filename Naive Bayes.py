#!/usr/bin/env python
# coding: utf-8

# In[2]:


import random
import functions
from sklearn import preprocessing

test_size = 80
test_list = random.sample(range(0, 500), test_size)
test_list.sort()


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
        test_row = int(round(test_row * 10))
        
        if index_test_list < test_size and test_list[index_test_list] == index_row_list:
            test_matrix.append(train_row)
            test_label.append(test_row)
            index_test_list += 1
        else:
            train_matrix.append(train_row)
            train_label.append(test_row)
        index_row_list += 1
        


# # Naive Bayes

# In[4]:


import numpy as np
x = np.array(train_matrix)
x = preprocessing.scale(x) # normalize
y = np.array(train_label)
x_test = np.array(test_matrix)
x_test = preprocessing.scale(x_test) # normalize


# In[5]:


from sklearn.model_selection import KFold

def cross_val(data, label, model):
    kfold = KFold(n_splits = 5, shuffle = False)
    index = kfold.split(X = data, y = label)
    cross_val_error = []
    for train_index, test_index in index:
        model.fit(data[train_index], label[train_index]) # train
        y_pred = model.predict(data[test_index]) # predict
        error = functions.RSS(y_pred, label[test_index]) # error
        cross_val_error.append(error)
    exp_error = sum(cross_val_error)/len(cross_val_error)
    return exp_error


# In[7]:


from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()
print('Gaussian error:', cross_val(x, y, gnb) / 100)


# In[8]:


from sklearn.metrics import accuracy_score

gnb = GaussianNB()
kfold = KFold(n_splits = 5, shuffle = False)
index = kfold.split(X = x, y = y)
for train_index, test_index in index:
    gnb.fit(x[train_index], y[train_index]) # train
    y_pred = gnb.predict(x[test_index]) # predict
    accuracy = accuracy_score(list(y_pred), list(y[test_index]))
print('Gaussian accuracy', accuracy)


# In[9]:


from sklearn.naive_bayes import BernoulliNB

gnb = BernoulliNB()
print('Bernoulli error:', cross_val(x, y, gnb) / 100)


# In[10]:


from sklearn.metrics import accuracy_score

gnb = BernoulliNB()
kfold = KFold(n_splits = 5, shuffle = False)
index = kfold.split(X = x, y = y)
for train_index, test_index in index:
    gnb.fit(x[train_index], y[train_index]) # train
    y_pred = gnb.predict(x[test_index]) # predict
    accuracy = accuracy_score(list(y_pred), list(y[test_index]))
print('Bernoulli accuracy', accuracy)


# In[ ]:




