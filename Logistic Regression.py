#!/usr/bin/env python
# coding: utf-8

# In[19]:


#  Predicting Stock Market Direction Using Logistic Regression

#Objective: Use logistic regression to predict the direction of the S&P 500 (Up or Down) using historical daily return data from 2001–2005.


# In[7]:


import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.metrics import accuracy_score, confusion_matrix

data = pd.read_csv('smarket.csv', index_col=0)
data['Direction'] = data['Direction'].map({'Down': 0, 'Up': 1})
data.head()
#We load the dataset and convert the `Direction` column into binary labels:  
#- **0** for Down  
#- **1** for Up  
#This prepares the data for binary classification using logistic regression.


# In[8]:


train = data[data['Year'] < 2005]
test = data[data['Year'] == 2005]

print('Shape of train is', train.shape)
print('Shape of test is ', test.shape)


# In[9]:


#We split the dataset:
#- **Training set**: years 2001–2004 (998 samples)
#- **Test set**: year 2005 (252 samples)


# In[ ]:


X_train = train.iloc[:, 1:-2]  # Drop 'Today' and 'Direction'
X_train = sm.add_constant(X_train)
y_train = train['Direction']

X_test = test.iloc[:, 1:-2]
X_test = sm.add_constant(X_test)
y_test = test['Direction']

results = sm.Logit(y_train, X_train).fit()
pred_probs = results.predict(X_test)
pred_classes = (pred_probs > 0.5).astype(int)

cm = confusion_matrix(y_test, pred_classes)
acc = accuracy_score(y_test, pred_classes)

print("Confusion Matrix (All Predictors):")
print(cm)
print(f"Accuracy: {acc:.3f}")


# In[ ]:


### Logistic Regression Using All Predictors

We fit the model using all 6 predictors: `Lag1` to `Lag5` and `Volume`.

**Result**:
- The model achieves only **~48% accuracy**, which is worse than random guessing.
- This suggests that the model may be **overfitting** to the noise in the training data.


# In[16]:


predictors = ['Lag1', 'Lag2']

X_train = sm.add_constant(train[predictors])
X_test = sm.add_constant(test[predictors])

results = sm.Logit(y_train, X_train).fit()
pred_probs = results.predict(X_test)
pred_classes = (pred_probs > 0.5).astype(int)

cm = confusion_matrix(y_test, pred_classes)
acc = accuracy_score(y_test, pred_classes)

print("Confusion Matrix (Lag1 & Lag2 Only):")
print(cm)
print(f"Accuracy: {acc:.3f}")
### Logistic Regression Using Only Lag1 and Lag2

#Based on the initial model summary, only `Lag1` and `Lag2` had statistically significant coefficients.

#Result
# Accuracy improves to **~56%**
# Simpler model performs better by avoiding overfitting


# 
# 
# 

# In[18]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
data = pd.read_csv('smarket.csv',index_col=0)
data.head()
data.info()
corr = data.corr()
plt.figure(figsize = (8,8))
sns.heatmap(corr)
# plot 'Volume' against index
plt.figure(figsize = (12,6))
sns.scatterplot(data.index,data['Volume'])


# In[ ]:




