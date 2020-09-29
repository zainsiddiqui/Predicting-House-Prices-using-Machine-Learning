#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[5]:


data = pd.read_csv("housing.csv",delim_whitespace=True,header = None)


# In[3]:


data.head()


# In[4]:


col_name = ['CRIM', 'ZN' , 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']


# In[5]:


data.columns = col_name


# In[6]:


data.head()


# In[7]:


data.describe()


# In[8]:


sns.pairplot(data,size = 1.5)


# In[9]:


col_study = ['ZN', 'INDUS', 'NOX', 'RM']


# In[11]:


sns.pairplot(data[col_study], size=2.5);
plt.show()


# In[12]:


col_study = ['PTRATIO', 'B', 'LSTAT', 'MEDV']


# In[14]:


sns.pairplot(data[col_study], size=2.5);
plt.show()


# In[17]:


plt.figure(figsize = (15,10))
sns.heatmap(data.corr(),annot =True)


# In[37]:


X = data['RM'].values.reshape(-1,1)


# In[38]:


y = data['MEDV'].values
from sklearn.model_selection import train_test_split
train_x,test_x,train_y,test_y = train_test_split(X,y,test_size =0.3,random_state = 101)


# In[39]:


from sklearn.linear_model import LinearRegression


# In[40]:


lrm = LinearRegression()


# In[41]:


lrm.fit(train_x,train_y)


# In[42]:


lrm.coef_


# In[43]:


lrm.intercept_


# In[44]:


pred = lrm.predict(test_x)


# In[45]:


from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score


# In[46]:


print("MAE",mean_absolute_error(test_y,pred))
print("MSE",mean_squared_error(test_y,pred))
print("R2_SCORE",r2_score(test_y,pred))
print("RMSE",np.sqrt(mean_absolute_error(test_y,pred)))


# In[37]:


sns.jointplot(test_y,pred)


# ## ROBUST REGRESSION(RANDOM SAMPLE COSENCUS)

# In[1]:


from sklearn.linear_model import RANSACRegressor


# In[7]:


col_name = ['CRIM', 'ZN' , 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
data.columns = col_name
data.head()


# In[48]:


X = data['RM'].values.reshape(-1,1)
y = data['MEDV'].values


# In[49]:


ransac = RANSACRegressor()


# In[50]:


ransac.fit(train_x,train_y)


# In[51]:


inlier_mask = ransac.inlier_mask_
outlier_mask = np.logical_not(inlier_mask)


# In[52]:


line_x = np.arange(3,10,1)
line_y_ransac = ransac.predict(line_x.reshape(-1,1))


# In[53]:


ransac.estimator_.coef_


# In[54]:


ransac.estimator_.intercept_


# In[55]:


y_train_pred = lrm.predict(train_x)


# In[56]:


y_test_pred = lrm.predict(test_x)


# In[57]:


plt.scatter(y_train_pred,y_train_pred - train_y)
plt.scatter(y_test_pred,y_test_pred-test_y)


# In[59]:


print("MAE", mean_absolute_error(y_test_pred,test_y))


# ## RIDGE MODEL

# L2 Penalized Model 
# address soome of problems of ordinary least squares by imposing a penality on size of coefficent
# can not zero out coefficents you either end up including all the coefficents in the model or none of them

# In[60]:


from sklearn.linear_model import Ridge


# In[62]:


ridge_model = Ridge(alpha = 1,normalize = True)


# In[63]:


ridge_model.fit(train_x,train_y)


# In[64]:


ridge_model_pred = ridge_model.predict(test_x)


# In[65]:


ridge_model.coef_


# In[66]:


ridge_model.intercept_


# ## LASSO REGRESSION

# A Linear model that estimates sparse coefficents
# used when multiple features are corelated but lasso pick random from these features and works on.
# It does both parameter shrinkage and variables selection automatically

# In[67]:


from sklearn.linear_model import Lasso


# In[68]:


Lasso = Lasso()


# In[69]:


Lasso.fit(train_x,train_y)


# In[71]:


pred = Lasso.predict(test_x)


# In[73]:


Lasso.intercept_


# In[74]:


Lasso.coef_


# ## ELASTIC NET

# A Linear Regression Model trained with L1 and L2 prior as regularizer
#  useful when multiple features are corelated with each other and it pick both features and works on
#  If some of your covariates are highly corelated you use elastic net

# In[75]:


from sklearn.linear_model import ElasticNet


# In[76]:


d = ElasticNet()


# In[77]:


d.fit(train_x,train_y)


# In[78]:


d.intercept_


# In[79]:


d.coef_


# ## LOGISTIC REGRESSION

# Used for classification rather regression
