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


# # DBSCAN

# ## DBSCAN - scaling

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


# ## DBSCAN - choose the number of cluster - PCA

# In[71]:


from sklearn.cluster import DBSCAN

dbscan = DBSCAN(eps=0.5, min_samples=2*x.shape[1]).fit(standard_x)# 0.010990990990990991 #eps=0.5, min_samples=2*x.shape[1]
y_pred = dbscan.labels_


cluster_center = dbscan.components_
cluster_center.shape
x.shape


# In[72]:


#check result using PCA
from mpl_toolkits.mplot3d import Axes3D

pca = PCA(n_components=2)
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


# ## DBSCAN - silhouette_score
# #https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html

# In[84]:


from sklearn.metrics import silhouette_samples

dbscan = DBSCAN(eps=0.5, min_samples=2*x.shape[1]).fit(standard_x)#
y_pred = dbscan.labels_

dbscan_silhouette_samples = silhouette_samples(x,y_pred)
dbscan_silhouette_samples


# In[115]:


from sklearn.metrics import silhouette_score

dbscan = DBSCAN(eps=0.5, min_samples=2*x.shape[1]).fit(standard_x)#standard_x , standard_x[:,[2,3]]
y_pred = dbscan.labels_

dbscan_silhouette_score = silhouette_score(x,y_pred)
dbscan_silhouette_score


# In[74]:




silhouette_score_list = []
eps_range = np.linspace(0.01,1,1000)

for eps_val in eps_range:
    dbscan = DBSCAN(eps=eps_val, min_samples=2*x.shape[1]).fit(standard_x)
    y_pred = dbscan.labels_
    
    if len(np.unique(y_pred)) > 1:
        dbscan_silhouette_score = silhouette_score(x,y_pred)
        silhouette_score_list.append(dbscan_silhouette_score)
    else:
        silhouette_score_list.append(-1)
    


plt.plot(eps_range, silhouette_score_list, 'bx-')
plt.xlabel('eps')
plt.ylabel('silhouette_score')
plt.title('DBSCAN silhouette_score')
plt.show()



# ## DBSCAN - silhouette_score - hyperparameter

# In[51]:


from mpl_toolkits.mplot3d import Axes3D

silhouette_score_list = []
eps_range = np.linspace(0.01,1,1000)
min_samples_range = range(1,5*x.shape[1],1)

eps_list = list()
min_samples_list = list()

for eps_val in eps_range:
    for min_samples_val in min_samples_range:
    
        dbscan = DBSCAN(eps=eps_val, min_samples=min_samples_val).fit(standard_x)
        y_pred = dbscan.labels_

        if len(np.unique(y_pred)) > 1:
            dbscan_silhouette_score = silhouette_score(x,y_pred)
            silhouette_score_list.append(dbscan_silhouette_score)
        else:
            silhouette_score_list.append(-1)
        eps_list.append(eps_val)
        min_samples_list.append(min_samples_val)
    

#3D plot
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(min_samples_list,eps_list, silhouette_score_list)#,c=y_pred
plt.xlabel('min_samples')
plt.ylabel('eps')
plt.title('DBSCAN silhouette_score')
plt.show()


# In[58]:


silhouette_score_np = np.array(silhouette_score_list)
silhouette_score_np.argmax()
silhouette_score_list[1]


# In[59]:


eps_range[1]


# In[60]:


min_samples_range[1]


# ## DBSCAN - make cluster

# In[ ]:





# ## DBSCAN - Feature Importance - Feature Selection

# In[ ]:


dbscan = DBSCAN(eps=0.5, min_samples=2*x.shape[1]).fit(standard_x)#
y_pred = dbscan.labels_


# In[112]:


silhouette_score_list = []
feature_range = range(0,standard_x.shape[1])

for i in feature_range:
    current_x = standard_x[:,i].reshape(-1, 1)
    
    dbscan = DBSCAN(eps=0.5, min_samples=2).fit(current_x)# min_samples=2*x.shape[1]
    y_pred = dbscan.labels_
    
    if len(np.unique(y_pred)) > 1:
        dbscan_silhouette_score = silhouette_score(x,y_pred)
        silhouette_score_list.append(dbscan_silhouette_score)
    else:
        silhouette_score_list.append(-1)
    print(i)
    


plt.plot(feature_range, silhouette_score_list, 'bx-')
plt.xlabel('feature')
plt.ylabel('silhouette_score')
plt.title('DBSCAN silhouette_score')
plt.show()


# In[110]:


standard_x.shape[1]


# In[ ]:




