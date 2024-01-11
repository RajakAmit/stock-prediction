#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')
# here we are importing important libraries


# In[4]:


#pip install --ignore-installed --upgrade tensorflow


# In[5]:


#!pip install pandas_datareader


# In[6]:


#!pip install keras
#!pip install tensorflow


# In[ ]:





# In[7]:


import math
#import tensorflow as tf
import pandas_datareader as web
import matplotlib.dates as mdates
plt.style.use("fivethirtyeight")
sns.set_style('whitegrid')
get_ipython().run_line_magic('matplotlib', 'inline')
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense,LSTM

import matplotlib.pyplot as plt


# In[8]:


data = pd.read_csv('E:\stock\TESLA.csv')


# In[9]:


data


# In[10]:


data.describe()


# In[11]:


data.info()


# In[12]:


data.isnull().sum()


# In[13]:


data.set_index('Date', inplace=True)


# In[14]:


# here we are visualising of closing price
data['Adj Close'].plot()
plt.title('Adjusted Close Prices Over Time')
plt.xlabel('Date')
plt.ylabel('Adj Close Price')
plt.grid(True)
plt.show()


# In[15]:


#Creating a new dataframe with only the 'Close' column
data = data.filter(['Adj Close'])
#Converting the dataframe to a numpy array
dataset = data.values
#Get /Compute the number of rows to train the model on
training_data_len = math.ceil( len(dataset) *.8)
training_data_len


# In[16]:


#features = ['Open','High','Low','Volume']
#output_var = pd.DataFrame(data['Adj Close'])


# In[17]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)
scaled_data


# In[18]:


#Creating the scaled training data set
train_data = scaled_data[0:training_data_len  , : ]
#Spliting the data into x_train and y_train data sets
x_train=[]
y_train = []
for i in range(60,len(train_data)):
    x_train.append(train_data[i-60:i,0])
    y_train.append(train_data[i,0])
    if i<= 61:
        print(x_train)
        print(y_train)
        print()


# In[19]:


#Here we are Converting x_train and y_train to numpy arrays
x_train, y_train = np.array(x_train), np.array(y_train)
#X_test, y_train = np.array(X_test), np.array(y_train)


# In[20]:


# Here we are reshaping the data into the shape accepted by the LSTM
x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))
#X_test = testX.reshape(X_test.shape[0], 1, X_test.shape[1])


# In[21]:


#now we are Building the LSTM network model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True,input_shape=(x_train.shape[1],1)))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dense(units=25))
model.add(Dense(units=1))


# In[22]:


# here we are Compiling the model
model.compile(optimizer='adam', loss='mean_squared_error')


# In[23]:


# here we are training the model
#model.fit(x_train, y_train, batch_size=1, epochs=200,verbose=1, shuffle=False, validation_data=(X_test, y_test))


# In[24]:


# Model Training
#history = model.fit(X_train, y_train, epochs=200, batch_size=4, verbose=1, shuffle=False, validation_data=(X_test, y_test))


# In[25]:


# here we are testing data set
test_data = scaled_data[training_data_len - 60: , : ]
#Creating the x_test and y_test data sets
x_test = []
y_test =  dataset[training_data_len : , : ] #Get all of the rows from index 1603 to the rest and all of the columns (in this case it's only column 'Close'), so 2003 - 1603 = 400 rows of data
for i in range(60,len(test_data)):
    x_test.append(test_data[i-60:i,0])


# In[26]:


# here we are converting x_test to a numpy array  
x_test = np.array(x_test)


# In[27]:


# here we are reshaping the data into the shape accepted by the LSTM  
x_test = np.reshape(x_test, (x_test.shape[0],x_test.shape[1],1))


# In[28]:


# here we are training the model
model.fit(x_train, y_train, batch_size=1, epochs=200,verbose=1, shuffle=False, validation_data=(x_test, y_test))


# In[30]:


print(x_train.shape)
print(x_test.shape)


# In[31]:


# now we are getting the models predicted price values
predictions = model.predict(x_test) 
predictions = scaler.inverse_transform(predictions)#Undo scaling


# In[32]:


# here we are calculaing the value of RMSE 
rmse=np.sqrt(np.mean(((predictions- y_test)**2)))
rmse


# In[33]:


# Plot the data
train = data[:training_data_len]
valid = data[training_data_len:]
valid['Predictions'] = predictions
# Visualize the data
plt.figure(figsize=(16,6))
plt.title('Model')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price USD ($)', fontsize=18)
plt.plot(train['Adj Close'])
plt.plot(valid[['Adj Close', 'Predictions']])
plt.legend(['Train', 'Valid', 'Predictions'], loc='lower right')
plt.show()


# In[34]:


print(valid)


# In[ ]:




