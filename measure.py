#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve


# In[3]:


actual_a=[1 for i in range(10)] + [0 for i in range(10)]
predicted_a=[1 for i in range(9)]+[0,1,1] +[0 for i in range(8)]
print(actual_a)
print(predicted_a)


# In[15]:


def my_confusion_matrix(actual,predicted):
    TP=len([a for a, p in zip(actual,predicted) if a==p and p==1])
    TN=len([a for a, p in zip(actual,predicted) if a==p and p==0])
    FP=len([a for a, p in zip(actual,predicted) if a!=p and p==1])
    FN=len([a for a, p in zip(actual,predicted) if a!=p and p==0])
    return "[[{} {}] \n[{} {}]]".format(TP,FN,FP,TN)
print(my_confusion_matrix(actual_a,predicted_a))
print(confusion_matrix(actual_a,predicted_a))

    


# In[18]:


array=[[9,1],
      [2,8]]
df_cm=pd.DataFrame(array,range(2),range(2))
sns.heatmap(df_cm,annot=True,cmap='Blues')
array=[[8,2],
      [9,1]]
df_cm=pd.DataFrame(array,range(2),range(2))
sns.heatmap(df_cm,annot=True,cmap='Oranges_r')


# In[19]:


def my_accuracy_score(actual,predicted):
    TP=len([a for a,p in zip(actual,predicted) if a==p and p==1])
    TN=len([a for a,p in zip(actual,predicted) if a==p and p==0])
    FP=len([a for a,p in zip(actual,predicted) if a!=p and p==1])
    FN=len([a for a,p in zip(actual,predicted) if a!=p and p==0])
    return (TP+TN)/(TP+TN+FP+FN)
print(my_accuracy_score(actual_a,predicted_a))
print(accuracy_score(actual_a,predicted_a))


# In[20]:


def my_precision_score(actual,predicted):
    TP=len([a for a,p in zip(actual,predicted) if a==p and p==1])
    FP=len([a for a,p in zip(actual,predicted) if a!=p and p==1])
    return TP/(TP+FP)
print(my_precision_score(actual_a,predicted_a))
print(precision_score(actual_a,predicted_a))


# In[21]:


def my_recall_score(actual,predicted):
    TP=len([a for a,p in zip(actual,predicted) if a==p and p==1])
    FN=len([a for a,p in zip(actual,predicted) if a!=p and p==0])
    return TP/(TP+FN)
print(my_recall_score(actual_a,predicted_a))
print(recall_score(actual_a,predicted_a))


# In[22]:


def my_f1_score(actual,predicted):
    x=my_precision_score(actual_a,predicted_a)
    y=my_recall_score(actual_a,predicted_a)
    return (2*x*y)/(x+y)
print(my_f1_score(actual_a,predicted_a))
print(f1_score(actual_a,predicted_a))


# In[24]:


prec,rec,_ =precision_recall_curve(actual_a,predicted_a)
plt.step(rec,prec,color='g',alpha=0.2,where='post')
plt.fill_between(rec,prec,color='g',alpha=0.2,step='post')
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.0])
plt.show()


# In[25]:


print(roc_auc_score(actual_a,predicted_a))
fpr,tpr,_ =roc_curve(actual_a,predicted_a)
# plt.figure()
plt.plot(fpr,tpr,color='orange',lw=2,label="ROC_CURVE")
plt.plot([0,1],[0,1],color='navy',lw=2,linestyle='--')
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.0])
plt.show()



# In[ ]:




