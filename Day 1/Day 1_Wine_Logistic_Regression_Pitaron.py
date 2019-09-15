#!/usr/bin/env python
# coding: utf-8

# # wine logistic regression

# In[1]:


#import
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn import datasets

get_ipython().run_line_magic('config', 'IPCompleter.greedy=True #autocomplete jupyter')


# In[2]:


# import some data to play with
wine = datasets.load_wine()
x = wine.data
y = wine.target

target_names = wine.target_names
features_names = wine.feature_names

mask_2 = y!=2
x = x[mask_2]
y = y[mask_2]

x_data_frame = pd.DataFrame(x,columns = features_names)


# In[3]:


x_data_frame.info()


# In[4]:


target_names


# # train test val split

# In[5]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2)#random_state


# # Applied Logistic Regression

# In[6]:


#Create the Classifier
logreg = LogisticRegression()

#Fit the Classifier
logreg.fit(x_train, y_train)

#predict
y_pred = logreg.predict(x_test)

#Predict Proba
y_proba = logreg.predict_proba(x_test)

#score calculation
y_score = logreg.score(x,y)*100

print("prediction on test set")
print(y_pred)
print("\n")

print("probability prediction on test set")
print(y_proba)
print("\n")

print("score of test prediction")
print(y_score)


# # Logistic Regression - Parameters - max_iter parameter

# In[7]:


# Example With - max_iter parameter

score_list = list()
max_value = 20

for i in range(max_value):
    #Create the Classifier
    logreg = LogisticRegression(max_iter=i)

    #Fit the Classifier
    logreg.fit(x_train, y_train)

    #predict
    y_pred = logreg.predict(x_test)
    
    y_score = logreg.score(x_test,y_test)*100
    score_list.append(y_score)
    
plt.plot([i for i in range(max_value)],score_list)
plt.xlabel("number of iteration")
plt.ylabel("score")
plt.show()


# In[ ]:




