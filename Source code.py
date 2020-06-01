#!/usr/bin/env python
# coding: utf-8

# In[159]:


import pandas as pd
file1 = r'D:\Work\Projects\Store Transaction Imputation\Hackathon_Working_Data.csv'
df = pd.read_csv(file1)

print ("\nFirst 5 rows:\n",df.head())
print ("\nLast 5 rows:\n",df.tail())


# In[160]:


y_data = df['VALUE']


# In[161]:


x_data = df.drop('VALUE', axis=1)


# In[162]:


print(y_data.head())


# In[163]:


print(x_data.head())


# In[164]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.4, random_state=0)


# In[165]:


print('No. of test samples:', x_test.shape[0])
print('No. of train samples:', x_train.shape[0])


# In[166]:


from sklearn.linear_model import LinearRegression
lr = LinearRegression()


# In[167]:


lr.fit(x_train[['PRICE']], y_train)


# In[168]:


lr.score(x_test[['PRICE']], y_test)


# In[169]:


lr.score(x_train[['PRICE']], y_train)


# In[170]:


x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.69, random_state=0)


# In[171]:


lr.fit(x_train[['PRICE']], y_train)
lr.score(x_test[['PRICE']], y_test)


# In[172]:


lr.score(x_train[['PRICE']], y_train)


# In[173]:


from sklearn.model_selection import cross_val_score
Rcross = cross_val_score(lr, x_data[['PRICE']], y_data, cv=4)


# In[174]:


print("The mean of the folds are", Rcross.mean(), "and the standard deviation is: ", Rcross.std())


# In[175]:


from sklearn.model_selection import cross_val_predict


# In[176]:


yhat = cross_val_predict(lr, x_data[['PRICE']], y_data, cv=4)


# In[177]:


yhat[0:5]


# In[178]:


lr.fit(x_train[['PRICE','BILL_AMT','QTY']], y_train)


# In[179]:


yhat_train = lr.predict(x_train[['PRICE','BILL_AMT','QTY']])


# In[180]:


yhat_train[0:5]


# In[181]:


yhat_test = lr.predict(x_test[['PRICE','BILL_AMT','QTY']])


# In[182]:


yhat_test[0:5]


# In[183]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[184]:


Title = 'Dist plot of Predict value using training data Vs Training data distribution'


# In[185]:


def DistributionPlot(Redf, Bluef, Redn, Bluen, Title):
    width =12
    height=10
    plt.figure(figsize=(width,height))
    ax1 = sns.distplot(Redf, hist=False, color='r', label=Redn)
    ax1 = sns.distplot(Bluef, hist=False, color='b', label=Bluen)
    plt.title(Title)
    plt.xlabel('Value')
    plt.ylabel('Features')
    plt.show()
    plt.close()


# In[186]:


DistributionPlot(y_train, yhat_train, "Actual", "Predicted", Title)


# In[187]:


Title = 'Dist plot of Predict value using testing data Vs Testing data distribution'
DistributionPlot(y_test, yhat_test, "Actual", "Predicted", Title)


# In[188]:


from sklearn.preprocessing import PolynomialFeatures
pr = PolynomialFeatures(degree=2)
x_train_pr = pr.fit_transform(x_train[['PRICE']])
x_test_pr = pr.fit_transform(x_test[['PRICE']])


# In[189]:


pr


# In[190]:


poly = LinearRegression()
poly.fit(x_train_pr, y_train)


# In[191]:


yhat = poly.predict(x_test_pr)


# In[192]:


print("Predicted values:", yhat[0:4])
print("Actual values:", y_test[0:4].values)


# In[193]:


import numpy as np
def PollyPlot(xtrain, xtest, ytrain, ytest, lr, poly_transform):
    width = 12
    height = 10
    plt.figure(figsize=(width,height))
    xmax = max([xtrain.values.max(), xtest.values.max()])
    xmin = min([xtrain.values.min(), xtest.values.min()])
    x = np.arange(xmin, xmax, 0.1)
    plt.plot(xtrain, ytrain, 'ro', label = 'Training data')
    plt.plot(xtest, ytest, 'go', label = 'Test data')
    plt.plot(x, lr.predict(poly_transform.fit_transform(x.reshape(-1,1))), label='Predicted function')
    plt.ylim([-10000,60000])
    plt.ylabel('Price')
    plt.legend()


# In[194]:


PollyPlot(x_train[['PRICE']], x_test[['PRICE']], y_train, y_test, poly, pr)


# In[195]:


poly.score(x_train_pr, y_train)


# In[196]:


poly.score(x_test_pr, y_test)

