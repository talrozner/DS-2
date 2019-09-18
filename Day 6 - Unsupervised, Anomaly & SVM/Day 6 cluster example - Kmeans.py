#!/usr/bin/env python
# coding: utf-8

# # Steps of clustering research
1. Selecting Algorithms
2. Scaling
3. Clustering
4. PAC & Checking the results
# In[1]:


#enable autocomplete
get_ipython().run_line_magic('config', 'IPCompleter.greedy=True')

#load packages
import sys #access to system parameters
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets

iris = datasets.load_iris()
x = iris.data
y = iris.target

x.shape


# # KMeans

# ## KMeans scaling

# In[2]:


from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

standard_scaler = StandardScaler()
minmax_scaler = MinMaxScaler()
pca_scaler = PCA(n_components=4)

standard_scaler.fit(x)
minmax_scaler.fit(x)
pca_scaler.fit(x)

standard_x = standard_scaler.transform(x)
minmax_x = minmax_scaler.transform(x)
pca_x = pca_scaler.transform(x)


# In[3]:


plt.hist(standard_x)
plt.show()


# In[4]:


plt.hist(minmax_x)
plt.show()


# In[5]:


plt.hist(pca_x)
plt.show()


# ## KMeans - choose the number of cluster - Elbow method 

# In[6]:


#https://blog.cambridgespark.com/how-to-determine-the-optimal-number-of-clusters-for-k-means-clustering-14f27070048f
from sklearn.cluster import KMeans

Sum_of_squared_distances = []
K = range(1,15)
for k in K:
    km = KMeans(n_clusters=k)
    km = km.fit(standard_x)
    Sum_of_squared_distances.append(km.inertia_)

plt.plot(K, Sum_of_squared_distances, 'bx-')
plt.xlabel('k')
plt.ylabel('Sum_of_squared_distances')
plt.title('Elbow Method For Optimal k')
plt.show()


# In[7]:


Diff_Sum_of_squared_distances = pd.Series(Sum_of_squared_distances)
Diff_Sum_of_squared_distances = Diff_Sum_of_squared_distances.diff().abs()#.diff().abs()
Diff_Sum_of_squared_distances.dropna(inplace=True)


plt.plot(K[1:], Diff_Sum_of_squared_distances, 'bx-')
plt.xlabel('k')
plt.ylabel('Diff - Sum_of_squared_distances')
plt.title('Diff -Elbow Method For Optimal k')
plt.show()

#Diff_Sum_of_squared_distances


# ## KMeans - make cluster

# In[8]:


km = KMeans(n_clusters=6)
km = km.fit(standard_x)
y_pred = km.predict(standard_x)
cluster_center = km.cluster_centers_


# ## KMeans - PAC & Checking the results

# In[9]:


#check result using PCA
from mpl_toolkits.mplot3d import Axes3D

pca = PCA(n_components=3)
pca.fit(standard_x)  
x_pca = pca.transform(standard_x)

pca_cluster_center = PCA(n_components=2)
pca_cluster_center.fit(cluster_center)  
pca_cluster_center = pca_cluster_center.transform(cluster_center)

#2D plot
plt.scatter(x_pca[:,0],x_pca[:,1],c=y_pred,alpha=1)
plt.show()

pca = PCA(n_components=3)
pca.fit(standard_x)  
x_pca = pca.transform(standard_x)

pca_cluster_center = PCA(n_components=3)
pca_cluster_center.fit(cluster_center)  
pca_cluster_center = pca_cluster_center.transform(cluster_center)

#3D plot
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(x_pca[:,0],x_pca[:,1],x_pca[:,2],c=y_pred)
plt.show()


# In[10]:


np.unique(y_pred)


# In[11]:


cluster_data = pd.DataFrame(x)
cluster_data['cluster'] = y_pred


# In[12]:


cluster_data.head(n=7)


# In[13]:


group_0 = cluster_data[cluster_data.cluster == 0]


# In[14]:


group_0.head()


# In[ ]:




