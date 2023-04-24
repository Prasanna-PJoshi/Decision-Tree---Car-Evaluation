#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree


# In[2]:


df = pd.read_csv('car_evaluation.csv', header=None)


# In[3]:


df.info()


# In[4]:


df.head()


# In[5]:


col_names= ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']
df.columns=col_names
df.head()


# In[6]:


df.isna().sum()


# In[7]:


df['buying'].value_counts()


# In[8]:


df['maint'].value_counts()


# In[9]:


df['lug_boot'].value_counts()


# In[10]:


df['safety'].value_counts()


# In[11]:


df['class'].value_counts()


# In[12]:


cat = df.select_dtypes(include = ['object'])
cat.head()


# In[13]:


pip install category_encoders


# In[14]:


import category_encoders as ce


# In[17]:


X=df.drop(['class'], axis=1)
Y=df['class']


# In[18]:


from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state = 30)


# In[19]:


encoder = ce.OrdinalEncoder(cols=['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety'])


X_train = encoder.fit_transform(X_train)

X_test = encoder.transform(X_test)


# In[20]:


model=DecisionTreeClassifier(criterion='entropy',
    max_depth=4,
    min_samples_split=2,
    min_samples_leaf=3,
    random_state=20,
    max_leaf_nodes=3)
model.fit(X_train,Y_train)


# In[21]:


model.score(X_train,Y_train)


# In[29]:


Y_pred=model.predict(X_test)


# In[23]:


from sklearn.metrics import accuracy_score


# In[31]:


accuracy_score(Y_test, Y_pred)


# In[32]:


Y_pred_train=model.predict(X_train)


# In[33]:


accuracy_score(Y_train, Y_pred_train)


# In[34]:


tree.plot_tree(model.fit(X_train,Y_train))


# In[ ]:




