#!/usr/bin/env python
# coding: utf-8

# # Logistic Regression

# In[87]:


#import
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn import datasets

get_ipython().run_line_magic('config', 'IPCompleter.greedy=True #autocomplete jupyter')


# In[88]:


# import some data to play with
iris = datasets.load_iris()
x = iris.data
y = iris.target

target_names = iris.target_names
features_names = iris.feature_names

mask_2 = y!=2
x = x[mask_2]
y = y[mask_2]

x_data_frame = pd.DataFrame(x,columns = features_names)


# In[89]:


x_data_frame.info()


# In[90]:


np.unique(y)


# # Split Training and Testing Data

# In[91]:


from IPython.display import Image
Image(filename='‏‏Test Harness.PNG')


# In[92]:


Image(filename='train_test_split.png')


# # train test split

# In[93]:


#https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
import numpy as np
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33)#random_state

print("All Features" + "\n")
print(x)
print("\n")

print("All Target" + "\n")
print(list(y))
print("\n")

print("train Features" + "\n")
print(x_train)
print("\n")

print("train Target" + "\n")
print(y_train)
print("\n")

print("test Features" + "\n")
print(x_test)
print("\n")

print("test Target" + "\n")
print(y_test)
print("\n")


# # train test val split

# In[94]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2)#random_state

print("All Features" + "\n")
print(x)
print("\n")

print("All Target" + "\n")
print(list(y))
print("\n")

print("train Features" + "\n")
print(x_train)
print("\n")

print("train Target" + "\n")
print(y_train)
print("\n")

print("test Features" + "\n")
print(x_test)
print("\n")

print("test Target" + "\n")
print(y_test)
print("\n")

print("val Features" + "\n")
print(x_val)
print("\n")

print("val Target" + "\n")
print(y_val)
print("\n")


# # Applied Logistic Regression

# In[95]:


#https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html

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


# # Logistic Regression - Parameters

# In[96]:


Image(filename='Logistic_regression_parameters.PNG')


# In[97]:


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


# # Logistic Coefficients

# In[98]:


print("Logistic Coefficients")
logreg.coef_


# In[99]:


logreg.coef_.shape


# In[101]:


print("Theta Shape")
theta = logreg.coef_
print(theta.shape)
print("\n")

print("Invers Theta Shape")
invers_theta = logreg.coef_.T
print(invers_theta.shape)
print("\n")

print("x shape")
print(x.shape)
print("\n")

print("y num of labels")
print(np.unique(y))
print("\n")

z = np.dot(x,theta.T)

h = 1/(1+np.exp(-z))

print("Hypothesis Shape")
print(h.shape)
print("\n")

print("Sample Hypothesis")
y_0 = h[99,:]
print(y_0)
print("\n")

print("Sample Proba Predict")
y_proba = logreg.predict_proba(x)
y_0_proba = y_proba[99,:]
print(y_0_proba)
print("\n")


# In[127]:


z_list = list(z.T)[0]
plt.figure()
plt.plot(z_list,y,"o")
plt.ylabel("Target (y)")
plt.xlabel("Z function")
plt.show()


# In[ ]:




