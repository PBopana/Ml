#!/usr/bin/env python
# coding: utf-8

# In[1]:


#imports
#KMeans
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


# In[2]:


df = pd.read_csv('https://raw.githubusercontent.com/saihaneesh26/ML_lab/main/datasets/ColourXY.csv')
df.head()


# In[3]:


df = df.drop(['color'],axis=1) # bcoz we are supposed to predict the cluster it belong and we dont know how many clusters are present initially
df.dropna(inplace=True)


# In[4]:


#scaling the data
scaler = StandardScaler()
df[ ['x','y'] ] = scaler.fit_transform(df[['x','y']])
df
#scaling req
#label encoder not req


# In[5]:


X = df['x'].to_numpy()
Y = df['y'].to_numpy() # conver to numpy
train_x,test_x,train_y,test_y = train_test_split(X,Y,test_size=0.3,shuffle=True,random_state = 2)


# In[6]:


train_data = []
for i in range(len(train_x)):
  train_data.append([train_x[i],train_y[i]])
train_data[:5]


# In[7]:


plt.scatter(train_x,train_y)
plt.xlabel("x")
plt.ylabel('y')
plt.show()


# In[8]:


wcss = []

distances = []
for k in range(1,11):
  model = KMeans(n_clusters = k)
  model.fit(train_data)
  centers = model.cluster_centers_
  total_sum = 0
  wcss.append(model.inertia_)
  labels = model.labels_


# In[9]:


wcss


# In[10]:


plt.plot([i for i in range(1,11)],wcss,'bx-')

plt.xlabel("K")
plt.ylabel("WCSS")
plt.show()


# In[11]:


random_point = np.array([150,80])
scaled_ = scaler.transform(random_point.reshape(1,-1))


# In[12]:


scaled_


# In[13]:


model = KMeans(n_clusters = 3)
model.fit(train_data)


# In[14]:


model.predict(scaled_.reshape(1,-1))


# In[15]:


plt.scatter(train_x,train_y)
plt.scatter(scaled_[0][0],scaled_[0][1],color='red')


# In[ ]:




