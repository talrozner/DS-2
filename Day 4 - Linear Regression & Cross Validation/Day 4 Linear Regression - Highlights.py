#!/usr/bin/env python
# coding: utf-8

# # pre-processing

# In[5]:


#enable autocomplete
get_ipython().run_line_magic('config', 'IPCompleter.greedy=True')

#load packages
import sys #access to system parameters
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#load data
file_path = r"titanic_train.csv"
data = pd.read_csv(file_path)
#complete missing age with median
data['Age'].fillna(data['Age'].median(), inplace = True)

#complete missing fare with median
data['Fare'].fillna(data['Fare'].median(), inplace = True)

#complete embarked with mode
data['Embarked'].fillna(data['Embarked'].mode()[0], inplace = True)

#complete Cabin with mode
data['Cabin'].fillna(data['Cabin'].mode()[0], inplace = True)

data['FamilySize'] = data['SibSp'] + data['Parch'] + 1

data['IsAlone'] = 1 #initialize to yes/1 is alone
data['IsAlone'].loc[data['FamilySize'] > 1] = 0

def my_regex(x):
    try:
        num = re.search(r'[0-9]+',x)
        return num.group(0)
    except:
        return 0
    return 

data['Cabin Num'] = data.Cabin.apply(lambda x : my_regex(x))    
#data.head()  

x = data[['Cabin Num','IsAlone','FamilySize','Parch','SibSp','Age','Survived']]

y = data.Fare


# # Data Exploration

# In[ ]:





# In[95]:


#https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
from sklearn.linear_model import LinearRegression

data['new age'] = np.exp(data['Age'])#**(1/2)
column_name = 'Age'

data['new fare'] = np.exp(data['Fare'])#**(1/2)
terget_column_name = 'new fare'

reg = LinearRegression().fit(data[column_name].values.reshape(-1, 1), data[terget_column_name].values.reshape(-1, 1))

print("Linear Regression Score")
print(reg.score(data[column_name].values.reshape(-1, 1), data[terget_column_name].values.reshape(-1, 1)))
print("\n")

print("Linear Regression coefficient")
print(reg.coef_)
print("\n")

print("Linear Regression intercept")
print(reg.intercept_ )
print("\n")


y_pred = reg.predict(data[column_name].values.reshape(-1, 1))


# In[96]:


#plot results

x_range = [i for i in range(100)]
y_line_list = list()
for i in x_range:
    y_line = reg.coef_[0]*i + reg.intercept_
    y_line_list.append(y_line)
    

plt.plot(data[column_name],data[terget_column_name],'o')
plt.plot(x_range,y_line_list)
plt.xlabel(column_name)
plt.ylabel(terget_column_name)
plt.legend([terget_column_name,'predicted ' + terget_column_name,'Regression Line'])
plt.show()


# In[97]:


sns.lmplot(x="Age", y="Fare", data=data,ci=99)


# In[98]:


data['new age'] = data['Age']**(1/5)
sns.lmplot(x="new age", y="Fare", data=data,ci=99)#,ci=99


# In[42]:


sns.jointplot(x="new age", y="Fare", data=data)


# In[59]:


f = plt.figure()
ax = f.add_subplot(1,1,1)
p = sns.regplot(x=data['Age'],y=data['Fare'],data=data,ax=ax)

#p.get_lines()[0].get_xdata()
#p.get_lines()[0].get_ydata()

#p.get_children()[1].get_paths()
p.ge


# # Correlation

# In[100]:


#correlation between features
corr = data.corr()
plt.figure()
sns.heatmap(corr, annot=True, annot_kws={"size": 40})
plt.suptitle("correlation between features", fontsize=40)
plt.show()


# In[103]:


#correlation between label and features
corr_label = data.corr()['Fare'].to_frame()
corr_label.sort_values('Fare',inplace=True)
plt.figure()
sns.heatmap(corr_label, annot=True, annot_kws={"size": 20})
plt.suptitle("correlation between label and features", fontsize=40)


# In[128]:


c = data.corr().abs()
s = c.unstack()
so = s.sort_values(kind="quicksort")
so_filter = so[so>0.5]

column_set = set()
for i,j in so_filter.index:
    column_set.add(i)
    column_set.add(j)


#print(column_set)
print(so.sort_values(ascending=False))


# In[129]:


print(so['Cabin Num'].sort_values(ascending=False))


# In[132]:


data['Cabin Num'].unique()


# # Feature Engineering

# In[135]:


print(so['FamilySize'].sort_values(ascending=False))


# In[138]:


data[['FamilySize','SibSp']].sort_values('FamilySize')


# In[ ]:





# # Feature Selection

# In[139]:


print(so['IsAlone'].sort_values(ascending=False))


# # Evaluation

# In[141]:


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)


# In[142]:


from sklearn.linear_model import LinearRegression

reg = LinearRegression().fit(x_train, y_train)

print("Linear Regression Score")
print(reg.score(x_test, y_test))
print("\n")

print("Linear Regression coefficient")
print(reg.coef_)
print("\n")

print("Linear Regression intercept")
print(reg.intercept_ )
print("\n")


y_pred = reg.predict(x_test)


# In[148]:


#https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_absolute_error.html
from sklearn.metrics import mean_absolute_error , mean_squared_error

mean_absolute_error(y_test, y_pred)


# In[149]:


mean_squared_error(y_test, y_pred)


# In[150]:


np.sqrt(mean_squared_error(y_test, y_pred))


# In[ ]:




