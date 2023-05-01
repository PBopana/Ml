#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
df=pd.read_csv('titanic1.csv')
print(df.head(2))


# In[2]:


df.dropna()
df.drop(['Name'],axis=1,inplace=True)
print(df.head(3))


# In[8]:


from sklearn.preprocessing import LabelEncoder,StandardScaler
encoder=LabelEncoder()
df['Sex']=encoder.fit_transform(df['Sex'])
scaler=StandardScaler()
df[['Age','Fare']]= scaler.fit_transform(df[['Age','Fare']])
print(df.head(3))


# In[9]:


from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB

x_train,x_test,y_train,y_test=train_test_split(df.drop(['Survived'],axis=1),df['Survived'],test_size=0.2)
nb=BernoulliNB()
nb.fit(x_train,y_train)
nb.classes_


# In[10]:


from sklearn.metrics import confusion_matrix, accuracy_score, f1_score

y_pred=nb.predict(x_test)
acc=accuracy_score(y_pred,y_test)
f=f1_score(y_pred,y_test)
print("Accuracy= ",acc ,"\nF1 score= ",f)


# In[11]:


print(y_test.shape)
labels=[0,1]
cm=confusion_matrix(y_pred,y_test,labels=labels)

import seaborn as sns
sns.heatmap(cm,annot=True,cmap="Blues")


# In[14]:


from sklearn.metrics import roc_curve, auc
prob=nb.predict_proba(x_test)
prob=prob[:,1]
fpr,tpr,_=roc_curve(y_test,prob)
print("AUC=",auc(fpr,tpr))
print("ROC curve")
sns.lineplot(x=fpr,y=tpr)


# In[ ]:




