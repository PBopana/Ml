#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
df=pd.read_csv('Advertising.csv')
print(df.head())
print("Dataframe shape=",df.shape)


# In[4]:


#plot media vs sales
import matplotlib.pyplot as plt
graphSheet=plt.figure(figsize=(15,10))
graphSheet.add_subplot(3,3,1)
plt.scatter(df['TV'],df['sales'])
plt.xlabel("Money spent on TV")
plt.ylabel("sales")
graphSheet.add_subplot(3,3,2)
plt.scatter(df['radio'],df['sales'], c="teal")
plt.xlabel("Money spent on radio")
plt.ylabel("sales")
graphSheet.add_subplot(3,3,3)
plt.scatter(df['newspaper'],df['sales'], c="orange")
plt.xlabel("Money spent on newspaper")
plt.ylabel("sales")


# In[15]:


#import statements
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,mean_absolute_error
import numpy as np


# In[42]:


def linReg(x,y):
    print((x+" vs "+y).center(40,'='))
    x_train,x_test,y_train,y_test=train_test_split(df[x],df[y],test_size=0.3)
    x_train=x_train.to_numpy().reshape(-1,1)
    x_test=x_test.to_numpy().reshape(-1,1)
    y_train=y_train.to_numpy().reshape(-1,1)
    y_test=y_test.to_numpy().reshape(-1,1)
    lr=LinearRegression()
    lr.fit(x_train,y_train)
    print("Coefficient= ",lr.coef_[0][0],"\nIntercept= ",lr.intercept_[0])
    pred=lr.predict(x_test)
    print("The linear regression line for {} vs {} is : Y={:.3} + {:.2}X".format(x,y,lr.intercept_[0],lr.coef_[0][0]))
    rmse=np.sqrt(mean_squared_error(y_test,pred))
    mse=mean_squared_error(y_test,pred)
    mae= mean_absolute_error(y_test,pred)
    print("RMSE = {}\n MSE= {}\n MAE= {}".format(rmse,mse,mae))
    plt.scatter(x_train,y_train)
    plt.scatter(x_test,y_test)
    plt.xlabel("money spent on"+x)
    plt.ylabel(y)
    plt.title(x+" vs "+y)
    plt.plot(x_test,pred, c='gold')
          


# In[44]:


sheet=plt.figure(figsize=(20,20))
sheet.add_subplot(2,2,1)
linReg('TV','sales')
sheet.add_subplot(2,2,2)
linReg('radio','sales')
sheet.add_subplot(2,2,3)
linReg('newspaper','sales')


# In[ ]:




