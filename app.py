import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data
from keras.models import load_model
import streamlit as st


start = '2010-01-01'
end = '2021-03-03'

st.title('Stock Trend Prediction')

user_input = st.text_input('Enter Stock Ticket', 'TSLA')
df = data.DataReader(user_input, 'yahoo',start,end)


#Describing Data

st.subheader('Data from 2010-2021')
st.write(df.describe())


#visualization

st.subheader('Opening Price vs Time Chart')
fig = plt.figure(figsize =(12,6))
plt.plot(df.Open)
st.pyplot(fig)

st.subheader('Opening Price vs Time Chart of 100 days Moving Average and 200 days Moving average')
ma100 = df.Open.rolling(100).mean()
ma200 = df.Open.rolling(200).mean()
fig = plt.figure(figsize =(12,6))
plt.plot(ma100, 'r')
plt.plot(ma200, 'g')
plt.plot(df.Open, 'b')
st.pyplot(fig)

st.subheader('Closing Price vs Time Chart')
fig = plt.figure(figsize =(12,6))
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('Closing Price vs Time Chart of 100 days Moving Average and 200 days Moving average')
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()
fig = plt.figure(figsize =(12,6))
plt.plot(ma100, 'r')
plt.plot(ma200, 'g')
plt.plot(df.Close, 'b')
st.pyplot(fig)


#splitting Data into Training and Testing

data_training =pd.DataFrame(df['Close'][0:int(len(df)*0.7)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.7): int(len(df))])

print(data_training.shape)
print(data_testing.shape)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))

data_training_array = scaler.fit_transform(data_training)

# x_train = []
# y_train = []

# for i in range(100, data_training_array.shape[0]):
#     x_train.append(data_training_array[i-100: i])
#     y_train.append(data_training_array[i-100: i])
    
# x_train, y_train =np.array(x_train), np.array(y_train)


#load my model
model = load_model('keras_model.h5')

#Testing Part

past_100_days = data_training.tail(100)
final_df =past_100_days.append(data_testing, ignore_index=True)
input_data =scaler.fit_transform(final_df)

x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100: i])
    y_test.append(input_data[i, 0])
    
x_test, y_test =np.array(x_test), np.array(y_test)

y_predicted = model.predict(x_test)

scaler =scaler.scale_
scale_factor = 1/scaler[0]
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor

#final graph
st.subheader('Prediction vs Original')
fig2=plt.figure(figsize=(12,6))
plt.plot(y_test,'b', label ='Original Price')
plt.plot(y_predicted,'r', label ='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)
