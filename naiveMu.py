#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import itertools
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris


# In[2]:


iris=load_iris()
print(iris.DESCR[:760])
print(iris.target_names)
print(iris.feature_names)


# In[3]:


X=iris.data
Y=iris.target


# In[4]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.4, random_state=1)


# In[5]:


from sklearn.naive_bayes import GaussianNB

gnb=GaussianNB()
gnb.fit(X_train,Y_train)


# In[6]:


y_pred = gnb.predict(X_test)


# In[7]:


print("Gaussian Naive Bayes model accuracy(in %):", metrics.accuracy_score(Y_test, y_pred)*100)


# In[8]:


print('Accuracy : ' , metrics.accuracy_score(Y_test, y_pred))
print('Precison : ' , metrics.precision_score(Y_test, y_pred, average="weighted"))
print('Recall Score : ' , metrics.recall_score(Y_test, y_pred, average="weighted"))
print('F1 Score : ' , metrics.f1_score(Y_test, y_pred, average="weighted"))
print('MCC : ' , metrics.matthews_corrcoef(Y_test, y_pred))


# In[10]:


sklearnconf=metrics.confusion_matrix(Y_test,y_pred)
print("Conf matrix in array form:\n",sklearnconf)


# In[11]:


labels=[1,0]
cm=metrics.confusion_matrix(Y_test,y_pred)
disp=metrics.ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()


# In[16]:


cm= metrics.confusion_matrix(Y_test,y_pred)
cm_df=pd.DataFrame(cm, index=['SETOSA','VERSICOLR','VIRGINICA'],columns=['SETOSA','VERSICOLR','VIRGINICA'])
plt.figure(figsize=(5,4))
sns.heatmap(cm_df, annot=True)
plt.title('Confusion Matrix')
plt.ylabel('Actal Values')
plt.xlabel('Predicted Values')
plt.show()


# In[ ]:




