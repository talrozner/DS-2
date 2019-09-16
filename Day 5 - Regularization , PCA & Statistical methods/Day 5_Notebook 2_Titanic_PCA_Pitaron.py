#!/usr/bin/env python
# coding: utf-8

# # PCA
# #https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
# #https://etav.github.io/python/scikit_pca.html

# In[181]:


#Imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import decomposition
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA
import seaborn as sb
get_ipython().run_line_magic('matplotlib', 'inline')

sb.set(font_scale=1.2,style="whitegrid") #set styling preferences

loan = pd.read_csv("creditcard.csv").sample(frac = .25) #read the dataset and sample 25% of it

#Data Wrangling
loan.replace([np.inf, -np.inf], np.nan) #convert infs to nans
loan = loan.dropna(axis = 1, how = 'any') #remove nans
loan = loan._get_numeric_data() #keep only numeric features

#Step 1: Standardize the Dataset
x = loan.drop('Class',axis=1).values #convert the data into a numpy array
x = scale(x);x

y = loan.Class 


# In[182]:


#https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca.fit(x)  
x_pca = pca.transform(x)

#2D plot
plt.scatter(x_pca[:,0],x_pca[:,1],c=y,cmap='viridis')
plt.show()

pca = PCA(n_components=3)
pca.fit(x)  
x_pca = pca.transform(x)

#https://stackoverflow.com/questions/1985856/how-to-make-a-3d-scatter-plot-in-python
#3D plot
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(x_pca[:,0],x_pca[:,1],x_pca[:,2],c=y,cmap='viridis')
plt.show()


# In[183]:


#Step 2: Create a Covariance Matrix
covar_matrix = PCA(n_components = 30) #we have 30 features

#Step 3: Calculate Eigenvalues
covar_matrix.fit(x)
variance = covar_matrix.explained_variance_ratio_ #calculate variance ratios

var=np.cumsum(np.round(covar_matrix.explained_variance_ratio_, decimals=3)*100)
var #cumulative sum of variance explained with [n] features

#Step 4, 5 & 6: Sort & Select

plt.ylabel('% Variance Explained')
plt.xlabel('# of Features')
plt.title('PCA Analysis')
plt.ylim(30,100.5)
plt.style.context('seaborn-whitegrid')


plt.plot(var,'o')
plt.show()


# In[184]:


variance


# In[185]:


var


# In[186]:


covar_matrix = PCA(n_components = 25)
covar_matrix.fit(x)


# In[187]:


covar_matrix.components_.shape


# In[188]:


covar_matrix.explained_variance_


# In[189]:


covar_matrix.explained_variance_ratio_


# In[190]:


var=np.cumsum(np.round(covar_matrix.explained_variance_ratio_, decimals=3)*100)
print(var) #cumulative sum of variance explained with [n] features


# In[191]:


plt.plot([i for i in range(len(var))],var)
plt.show()


# In[192]:


covar_matrix.singular_values_


# In[193]:


covar_matrix.mean_ 


# In[194]:


covar_matrix.n_components_ 


# In[195]:


covar_matrix.noise_variance_


# In[ ]:





# In[196]:


y.unique()

