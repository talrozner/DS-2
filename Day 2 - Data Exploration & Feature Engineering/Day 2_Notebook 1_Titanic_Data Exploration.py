#!/usr/bin/env python
# coding: utf-8

# # Data Exploration

# ## Data Info and Type

# In[1]:


#load packages
import sys #access to system parameters
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('config', 'IPCompleter.greedy=True #autocomplete jupyter')


# In[2]:


#load data
file_path = r"titanic_train.csv"
data = pd.read_csv(file_path)


# In[3]:


#data information
data.info()


# In[4]:


data.head()


# In[5]:


data.sample(10)


# In[6]:


data.describe()


# In[7]:


data.columns


# In[8]:


data.Cabin.nunique()


# In[9]:


data.Cabin.unique()


# In[10]:


data.Cabin.value_counts()


# ## Visualization

# In[43]:


data.info()


# In[50]:


#corelation plot
corr_data = data[['PassengerId','Survived','Pclass','Age','SibSp','Parch','Fare']]

f = plt.figure(figsize=(19, 15))
plt.matshow(corr_data.corr(), fignum=f.number)
plt.xticks(range(corr_data.shape[1]), corr_data.columns, fontsize=14, rotation=45)
plt.yticks(range(corr_data.shape[1]), corr_data.columns, fontsize=14)
cb = plt.colorbar()
cb.ax.tick_params(labelsize=14)
plt.title('Correlation Matrix', fontsize=16);

print("############corelation############")
corr_data.corr()


# In[11]:


#bar plot - Pclass
Cabin_frequncy = data['Pclass'].value_counts()
plt.figure()
plt.suptitle("Bar plot of Pclass", fontsize=40)
Cabin_frequncy.plot(kind='bar')
plt.xlabel("Value",fontdict={'fontsize': 40})
plt.ylabel("Frequency",fontdict={'fontsize': 40})
plt.show()


# In[12]:


#hist plot - Pclass
plt.hist(x = [data[data['Survived']==1]['Pclass'], data[data['Survived']==0]['Pclass']], 
         stacked=True, color = ['g','r'],label = ['Survived','Dead'])
plt.title('Pclass Histogram by Survival')
plt.xlabel('Pclass')
plt.ylabel('# of Passengers')
plt.legend()
plt.show()


# In[13]:


#hist plot - Pclass
plt.hist(x = [data[data['Survived']==1]['Fare'], data[data['Survived']==0]['Fare']], 
         stacked=True, color = ['g','r'],label = ['Survived','Dead'])
plt.title('Fare Histogram by Survival')
plt.xlabel('Fare')
plt.ylabel('# of Passengers')
plt.legend()
plt.show()


# In[37]:


#box plot - Age distibution
mask = ~np.isnan(data['Age']).values

plt.boxplot(x=data.loc[mask,'Age'], showmeans = True, meanline = True)
plt.title('Age Boxplot')
plt.ylabel('Age')
plt.show()


# In[15]:


#boxplot - sns
sns.boxplot(x = 'Pclass', y = 'Fare', hue = 'Survived', data = data)
plt.show()


# In[16]:


#barplot
sns.barplot(x = 'Embarked', y ='Survived', data=data)
plt.show()


# In[17]:


#barplot
sns.barplot(x = 'Embarked', y = 'Fare',hue = 'Survived', data=data)
plt.show()


# In[18]:


#pointplot
sns.pointplot(x = 'Embarked', y = 'Survived', data=data)
plt.show()


# In[19]:


#violinplot
sns.violinplot(x = 'Pclass', y = 'Age', hue = 'Survived', data = data, split = True)
plt.show()


# In[20]:


#how does class factor with sex & survival compare
sns.pointplot(x="Pclass", y="Survived", hue="Sex", data=data,
              palette={"male": "blue", "female": "pink"},
              markers=["*", "o"], linestyles=["-", "--"])


# In[21]:


#
e = sns.FacetGrid(data, col = 'Embarked')
e.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', ci=95.0, palette = 'deep')
plt.show()


# ## Find Outliers

# ## Making Decision
