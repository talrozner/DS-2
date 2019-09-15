#!/usr/bin/env python
# coding: utf-8

# In[61]:


#load packages
import sys #access to system parameters
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
get_ipython().run_line_magic('config', 'IPCompleter.greedy=True #autocomplete jupyter')


# In[2]:


#load data
file_path = r"titanic_train.csv"
data = pd.read_csv(file_path)


# # Sampling 

# # Formatting

# # Cleaning

# ## The 4 C's of Data Cleaning: Correcting, Completing, Creating, and Converting
In this stage, we will clean our data by
1) correcting aberrant values and outliers,
2) completing missing information, 
3) creating new features for analysis, and 
4) converting fields to the correct format for calculations and presentation.Correcting: Reviewing the data, there does not appear to be any aberrant or non-acceptable data inputs. In addition, we see we may have potential outliers in age and fare. However, since they are reasonable values, we will wait until after we complete our exploratory analysis to determine if we should include or exclude from the dataset. It should be noted, that if they were unreasonable values, for example age = 800 instead of 80, then it's probably a safe decision to fix now. However, we want to use caution when we modify data from its original value, because it may be necessary to create an accurate model.

Completing: There are null values or missing data in the age, cabin, and embarked field. Missing values can be bad, because some algorithms don't know how-to handle null values and will fail. While others, like decision trees, can handle null values. Thus, it's important to fix before we start modeling, because we will compare and contrast several models. There are two common methods, either delete the record or populate the missing value using a reasonable input. It is not recommended to delete the record, especially a large percentage of records, unless it truly represents an incomplete record. Instead, it's best to impute missing values. A basic methodology for qualitative data is impute using mode. A basic methodology for quantitative data is impute using mean, median, or mean + randomized standard deviation. An intermediate methodology is to use the basic methodology based on specific criteria; like the average age by class or embark port by fare and SES. There are more complex methodologies, however before deploying, it should be compared to the base model to determine if complexity truly adds value. For this dataset, age will be imputed with the median, the cabin attribute will be dropped, and embark will be imputed with mode. Subsequent model iterations may modify this decision to determine if it improves the model’s accuracy.

Creating: Feature engineering is when we use existing features to create new features to determine if they provide new signals to predict our outcome. For this dataset, we will create a title feature to determine if it played a role in survival.

Converting: Last, but certainly not least, we'll deal with formatting. There are no date or currency formats, but datatype formats. Our categorical data imported as objects, which makes it difficult for mathematical calculations. For this dataset, we will convert object datatypes to categorical dummy variables.
# ### correcting aberrant values and outliers

# In[3]:


data.info()


# In[4]:


#box plot - Fare distibution
mask = ~np.isnan(data['Fare']).values

plt.boxplot(x=data.loc[mask,'Fare'], showmeans = True, meanline = True)
plt.title('Fare Boxplot')
plt.ylabel('Fare')
plt.show()


# In[5]:


data.Fare.sort_values(ascending=False).head()


# In[6]:


data.Fare[data.Fare>263] = 263


# In[7]:


data.Fare.sort_values(ascending=False).head()


# In[8]:


#box plot - Fare distibution
mask = ~np.isnan(data['Fare']).values

plt.boxplot(x=data.loc[mask,'Fare'], showmeans = True, meanline = True)
plt.title('Fare Boxplot')
plt.ylabel('Fare')
plt.show()


# ### completing missing information

# In[9]:


data.info()


# In[10]:


fill_values = data.Age.mean()
print("mean age " + str(fill_values))
data.Age.fillna(fill_values,inplace=True)


# In[11]:


#complete Cabin with mode
data['Cabin'].fillna(data['Cabin'].mode()[0], inplace = True)


# In[12]:


data.info()


# In[13]:


#hist plot - Age
plt.hist(x = [data[data['Survived']==1]['Age'], data[data['Survived']==0]['Age']], 
         stacked=True, color = ['g','r'],label = ['Survived','Dead'])
plt.title('Age Histogram by Survival')
plt.xlabel('Age')
plt.ylabel('# of Passengers')
plt.legend()
plt.show()


# ### creating new features for analysis

# <font color='red'>#### The Consept of Feature Engineering</font>

# <font color='red'>#### Categorical Vs Continuous Features</font>

# <font color='red'>#### Create Features</font>

# In[26]:


data['FamilySize'] = data['SibSp'] + data['Parch'] + 1

data['IsAlone'] = 1 #initialize to yes/1 is alone
data['IsAlone'].loc[data['FamilySize'] > 1] = 0


# In[27]:


sns.barplot(x ='IsAlone', y ='Survived', data=data)
plt.show()


# In[28]:


#hist plot - FamilySize
plt.hist(x = [data[data['Survived']==1]['FamilySize'], data[data['Survived']==0]['FamilySize']], 
         stacked=True, color = ['g','r'],label = ['Survived','Dead'])
plt.title('FamilySize Histogram by Survival')
plt.xlabel('FamilySize')
plt.ylabel('# of Passengers')
plt.legend()
plt.show()


# <font color='red'>#### Grouping - create bins from categorical data</font>

# In[14]:


#hist plot - Age
plt.hist(x = [data[data['Survived']==1]['Age'], data[data['Survived']==0]['Age']], 
         stacked=True, color = ['g','r'],label = ['Survived','Dead'])
plt.title('Age Histogram by Survival')
plt.xlabel('Age')
plt.ylabel('# of Passengers')
plt.legend()
plt.show()


# In[15]:


import math

def roundup(x):
    return int(math.ceil(x / 10.0)) * 10

data['Age round'] = data.Age.apply(lambda x: roundup(x))
data.groupby('Age round')['Age round'].agg(['count'])


# In[16]:


#hist plot - Age
plt.hist(x = [data[data['Survived']==1]['Age round'], data[data['Survived']==0]['Age round']], 
         stacked=True, color = ['g','r'],label = ['Survived','Dead'])
plt.title('Age round Histogram by Survival')
plt.xlabel('Age round')
plt.ylabel('# of Passengers')
plt.legend()
plt.show()


# <font color='red'>#### Feature from Text</font>

# In[17]:


import re
def my_regex(x):
    try:
        num = re.search(r'[0-9]+',x)
        return num.group(0)
    except:
        return 0
    return 

data['Cabin Num'] = data.Cabin.apply(lambda x : my_regex(x))    
data['Cabin Num'].value_counts().head()
print(data['Cabin Num'].dtype)
data['Cabin Num'] = data['Cabin Num'].astype("int")
print(data['Cabin Num'].dtype)


# In[18]:


#hist plot - Cabin Num
plt.hist(x = [data[data['Survived']==1]['Cabin Num'], data[data['Survived']==0]['Cabin Num']], 
         stacked=True, color = ['g','r'],label = ['Survived','Dead'])
plt.title('Cabin Num Histogram by Survival')
plt.xlabel('Cabin Num')
plt.ylabel('# of Passengers')
plt.legend()
plt.show()


# <font color='red'>#### Scalling</font>

# In[19]:


#hist plot - Age
plt.hist(x = [data[data['Survived']==1]['Age round'], data[data['Survived']==0]['Age round']], 
         stacked=True, color = ['g','r'],label = ['Survived','Dead'])
plt.title('Age round Histogram by Survival')
plt.xlabel('Age round')
plt.ylabel('# of Passengers')
plt.legend()
plt.show()


# In[20]:


# Age scaled
survived_mask = (data.Survived==1).values
dead_mask = (data.Survived==0).values


age_ratio = data.loc[survived_mask,'Age round'].value_counts() / data.loc[dead_mask,'Age round'].value_counts()

age_ratio.fillna(0,inplace=True)
age_ratio

data['Age scaled'] = data['Age round'].map(age_ratio)

data[['Age round','Age scaled']].head()
index_list = list(age_ratio.sort_values(ascending=False).index.astype('O'))
index_list =(str(x) for x in index_list)


# In[21]:


#hist plot - Age scaled
plt.hist(x = [data[data['Survived']==1]['Age scaled'], data[data['Survived']==0]['Age scaled']], 
         stacked=True, color = ['g','r'],label = ['Survived','Dead'])
plt.title('Age scaled scaled Histogram by Survival')
plt.xlabel('Age scaled scaled')
#plt.xlabel(index_list)
plt.xticks(age_ratio.values, index_list)
plt.ylabel('# of Passengers')
plt.legend()
plt.show()


# In[22]:


age_ratio.sort_values(ascending=False)


# ### converting fields to the correct format for calculations and presentation

# <font color='red'>####Get Dummies </font>

# In[23]:


data_age_dummy = pd.get_dummies(data['Age round'])
data = data.join(data_age_dummy,lsuffix='_Age round')
data.drop('Age round',axis=1,inplace=True)
data.head()


# # Feature Selection - (Presentation)

# In[31]:


x = data[['Cabin Num','Age scaled',10,20,30,40,50,60,70,80,'FamilySize','Parch','SibSp','Age']]

y = data.Survived


# ## Chi Square Test

# In[74]:


from sklearn.feature_selection import SelectKBest,SelectPercentile, SelectFpr,SelectFdr,GenericUnivariateSelect,chi2,f_regression
x_chi = SelectKBest(chi2, k=2).fit_transform(x, y)

#Feature Selection - Filter Methods - Univariate feature selection
# Chi squared test - SelectKBest
x_chi = SelectKBest(chi2, k=2).fit_transform(x, y)

# Chi squared test - SelectPercentile
x_chi = SelectPercentile(chi2, percentile=2).fit_transform(x, y)

# Chi squared test - SelectFpr
x_chi = SelectFpr(chi2, alpha=0.001).fit_transform(x, y)

# Chi squared test - SelectFdr
x_chi = SelectFdr(chi2, alpha=0.001).fit_transform(x, y)

# Chi squared test - GenericUnivariateSelect - mode : {‘percentile’, ‘k_best’, ‘fpr’, ‘fdr’, ‘fwe’}
transformer = GenericUnivariateSelect(chi2, 'k_best', param=5)
x_chi = transformer.fit_transform(x, y)

chosen_columns = x.columns[transformer.get_support()]

x_chi_df = x[chosen_columns]

print(x.columns)

#correlation between label and features
corr_label = data.corr()['Survived'].to_frame()
corr_label.sort_values('Survived',inplace=True)
plt.figure()
sns.heatmap(corr_label, annot=True, annot_kws={"size": 20})
plt.suptitle("correlation between Survived and features", fontsize=40)


# ## Coefficient Correlation

# In[69]:


#correlation between features
corr = x_chi_df.corr()
plt.figure()
sns.heatmap(corr, annot=True, annot_kws={"size": 40})
plt.suptitle("correlation between features", fontsize=40)


columns = np.full((corr.shape[0],), True, dtype=bool)
for i in range(corr.shape[0]):
    for j in range(i+1, corr.shape[0]):
        if corr.iloc[i,j] >= 0.9:
            if columns[j]:
                columns[j] = False
selected_columns = x_chi_df.columns[columns]
x_chi_df = x_chi_df[selected_columns]



print(x_chi_df.head())


# # First Run of Model

# In[70]:


#Create the Classifier
logreg = LogisticRegression()

#Fit the Classifier
logreg.fit(x, y)

#predict
y_pred = logreg.predict(x)

#score calculation
y_score = logreg.score(x,y)*100


# In[71]:


print("score")
print(str(y_score) + "%")


# In[75]:


#Create the Classifier
logreg = LogisticRegression()

#Fit the Classifier
logreg.fit(x_chi_df, y)

#predict
y_pred = logreg.predict(x_chi_df)

#score calculation
y_score = logreg.score(x_chi_df,y)*100


# In[76]:


print("score")
print(str(y_score) + "%")


# In[ ]:




