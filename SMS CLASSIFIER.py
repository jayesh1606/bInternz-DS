#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[3]:


df = pd.read_csv('C:\\Users\\DELL\\mail_data.csv')


# In[4]:


print(df)      # ham --> no spam 


# In[5]:


data = df.where((pd.notnull(df)), '')


# In[7]:


data.head(10)      # prints top 10 lines of dataset


# In[9]:


data.info()


# In[10]:


data.shape


# In[ ]:


# 0 will show as spam and 1 will show ham(no spam) mails.


# In[21]:


data.loc[data['Category'] == 'spam','Category',] = 0
data.loc[data['Category'] == 'ham','Category',] = 1


# In[22]:


X = data['Message']
Y = data['Category']


# In[23]:


print(X)         # only prints messages


# In[24]:


print(Y)       # prints 0 and 1 as spam and ham respectively


# In[ ]:


# 0.2 specifies 80% training and 20% will be tested


# In[26]:


X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, random_state = 3)


# In[27]:


print(X.shape)
print(X_train.shape)
print(X_test.shape)


# In[28]:


print(Y.shape)
print(Y_train.shape)
print(Y_test.shape)


# In[32]:


feature_extraction = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)
X_train_features = feature_extraction.fit_transform(X_train)
X_test_features = feature_extraction.transform(X_test)

Y_train = Y_train.astype('int')
Y_test = Y_test.astype('int')


# In[33]:


print(X_train)


# In[34]:


print(X_train_features)


# In[35]:


model = LogisticRegression()


# In[36]:


model.fit(X_train_features,Y_train)


# In[37]:


prediction_on_training_data = model.predict(X_train_features)
accuracy_on_training_data = accuracy_score(Y_train,prediction_on_training_data)


# In[38]:


print('Accuracy on training data : ',accuracy_on_training_data)


# In[39]:


prediction_on_test_data = model.predict(X_test_features)
accuracy_on_test_data = accuracy_score(Y_test, prediction_on_test_data)


# In[40]:


print('Accuracy on test data : ',accuracy_on_test_data)


# In[58]:


input_mail = ["Hi Jayesh,Ready to become a better coder? Practicing for a programming job interview?"]

input_data_features = feature_extraction.transform(input_mail)

prediction = model.predict(input_data_features)

print(prediction)

if(prediction[0]==1):
    print('Ham mail')
else:
    print('Spam mail')


# In[ ]:




