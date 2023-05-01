#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
df=pd.read_csv('Advertising.csv')
print(df.head())
print("Dataframe shape=",df.shape)


# In[13]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

x_train,x_test,y_train,y_test=train_test_split(df.drop(['sales'],axis=1),df['sales'],test_size=0.25)
lr=LinearRegression()
lr.fit(x_train,y_train)


# In[6]:


print("Coefficient= ",lr.coef_ ,"\nIntercept= ", lr.intercept_)
print("Linear regression line of is: Y= {:.5} + {:.5}*TV + {:.5}*radio + {:.5}*newspapers".format(lr.intercept_,lr.coef_[0],lr.coef_[1],lr.coef_[2]))


# In[8]:


#print score
import numpy as np
from sklearn.metrics import mean_squared_error,mean_absolute_error

pred = lr.predict(x_test)

#Finding Root Mean Sqaured Error
rmse=np.sqrt(mean_squared_error(y_test,pred))
#Finiding Mean Absolute Error
mae = mean_absolute_error(y_test,pred)
#Finding Mean Squared Error
mse = mean_squared_error(y_test,pred)
#Printing all error matrics
print("Root Mean Square Srror= {}\nMean Absolute Error = {}\nMean Squared Error = {}".format(rmse,mae,mse))
print("Score=",lr.score(df.drop(['sales'],axis=1),df['sales']))


# In[9]:


def train2features(f1,f2):
    print(("Printing only two features:"+ f1+" vs "+f2).center(50,'='))
    x_train,x_test,y_train,y_test=train_test_split(df[[f1,f2]],df['sales'],test_size=0.25)
    lr=LinearRegression()
    lr.fit(x_train,y_train)
    print("Coefficients=",lr.coef_, "\nIntercepts=",lr.intercept_)
    print("Linear regression line is : Y= {:.5} + {:.5}*{} +{:.5}*{}".format(lr.intercept_, lr.coef_[0],f1,lr.coef_[1],f2))
   
    pred = lr.predict(x_test)
    #Finding Root Mean Sqaured Error
    rmse=np.sqrt(mean_squared_error(y_test,pred))
    #Finiding Mean Absolute Error
    mae = mean_absolute_error(y_test,pred)
    #Finding Mean Squared Error
    mse = mean_squared_error(y_test,pred)
    #Printing all error matrics
    print("Root Mean Square Srror= {}\nMean Absolute Error = {}\nMean Squared Error = {}".format(rmse,mae,mse))
    


# In[10]:


train2features('TV','radio')
train2features('TV','newspaper')
train2features('radio','newspaper')


# In[11]:


import seaborn as sns
sns.pairplot(df,x_vars=['TV','radio','newspaper'], y_vars='sales', kind='reg',aspect=1,height=7)


# In[12]:


sns.pairplot(df,x_vars=['TV','radio'], y_vars='sales', kind='reg',aspect=1,height=7)


# In[ ]:




