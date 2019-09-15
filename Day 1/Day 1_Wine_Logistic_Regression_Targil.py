#!/usr/bin/env python
# coding: utf-8

# # wine logistic regression

# In[2]:


#import
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn import datasets

get_ipython().run_line_magic('config', 'IPCompleter.greedy=True #autocomplete jupyter')


# In[3]:


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


# In[4]:


x_data_frame.info()


# In[5]:


target_names


# # train test val split

# In[ ]:





# # Applied Logistic Regression

# In[ ]:





# # Logistic Regression - Parameters - max_iter parameter

# In[ ]:




