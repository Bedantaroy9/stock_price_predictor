



import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import matplotlib.pyplot as plt
import datetime
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model

model = load_model("Stock_Predictions_Model3.keras", compile=False)



st.header('📈 Stock Market Predictor')

# User input
stock = st.text_input('Enter Stock Symbol', 'GOOG')

st.caption("💡 Enter any valid stock ticker from Yahoo Finance (e.g., AAPL, TSLA, INFY.NS, RELIANCE.NS)")

# Updated end date to today
start = '2012-01-01'
end = datetime.date.today()

# Download stock data
data = yf.download(stock, start, end)

st.subheader('🔍 Stock Data')

st.caption(f'Displaying historical stock data from **{start}** to **{end}** '
           f'for the selected stock symbol using Yahoo Finance.')

st.write(data)

# Train/test split (80/20)
data_train = pd.DataFrame(data.Close[0:int(len(data)*0.80)])
data_test = pd.DataFrame(data.Close[int(len(data)*0.80):len(data)])

# Scale the data
scaler = MinMaxScaler(feature_range=(0,1))
pas_100_days = data_train.tail(100)
data_test = pd.concat([pas_100_days, data_test], ignore_index=True)
data_test_scale = scaler.fit_transform(data_test)

st.subheader('📊 Price vs MA100')
st.caption('This graph shows the stock closing price alongside the 100-day moving average. '
           'MA100 is often used by traders to identify short-to-medium term trends.')
ma_100_days = data.Close.rolling(100).mean()
fig1 = plt.figure(figsize=(8,6))
plt.plot(ma_100_days, 'r')
plt.plot(data.Close, 'g')
plt.show()
st.pyplot(fig1)

# Prediction on test data
x = []
y = []
for i in range(100, data_test_scale.shape[0]):
    x.append(data_test_scale[i-100:i])
    y.append(data_test_scale[i,0])

x, y = np.array(x), np.array(y)
predict = model.predict(x)

# Reverse scaling
scale = 1/scaler.scale_
predict = predict * scale
y = y * scale

st.subheader('📉 Original Price vs Predicted Price ')
st.caption('Compares actual stock prices with model predictions on test data. '
           'This visual helps evaluate how well the model learned the price patterns.')
fig4 = plt.figure(figsize=(8,6))
plt.plot(predict, 'r', label='Predicted Price')
plt.plot(y, 'g', label='Original Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()
st.pyplot(fig4)


st.subheader('📍 Predicted Price for Tomorrow')
st.caption('This section shows the model’s predicted closing price for the next trading day.')

# Take the last 100 days for final prediction
recent_data = data.tail(100)[['Close']]
recent_data_scaled = scaler.transform(recent_data)

x_input = np.array(recent_data_scaled).reshape(1, 100, 1)
predicted_price = model.predict(x_input)

predicted_price = scaler.inverse_transform(predicted_price) ##X_original = X_scaled / scaler.scale_ + scaler.min_

next_day = datetime.date.today() + datetime.timedelta(days=1)
st.write(f'Predicted Closing Price on {next_day} ➜ **${predicted_price[0][0]:.2f}**')

# Final Prediction Graph
fig5 = plt.figure(figsize=(10, 6))

# Plot last 50 days of real data
recent_dates = data.index[-100:]
recent_prices = data.Close[-100:]
plt.plot(recent_dates, recent_prices, label='Last 100 Days Price', color='green')

# Plot predicted price for tomorrow
#Use pd.Timedelta and pandas.Timestamp when working with data.index
next_day = recent_dates[-1] + pd.Timedelta(days=1) #pd.Timedelta(days=1):Adds 1 calendar day
plt.scatter(next_day, predicted_price, color='red', label='Predicted Price (Tomorrow)', zorder=5)
#Withoutzorder=5, the red dot might:
# Format graph
plt.xlabel('Date')
plt.ylabel('Price')
plt.title(f'Prediction for {stock}')
plt.xticks(rotation=45) #names dates
plt.legend() # plt.legend() to show those labels in the plot.
plt.grid(True)

# Display on Streamlit
st.pyplot(fig5)