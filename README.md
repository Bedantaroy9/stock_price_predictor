# 📈 Stock Market Price Predictor

![stock](https://github.com/user-attachments/assets/f9453497-f188-4f1e-b9b7-174fadb1f251)



A real-time web application that predicts stock closing prices using an LSTM-based deep learning model. The app fetches historical stock data from Yahoo Finance, analyzes trends, and forecasts the next day’s closing price.

---

## 🔍 Features

-  Visualizes historical stock data with 100-day moving average (MA100)
-  Predicts stock prices using a 4-layer LSTM neural network
-  Compares actual vs predicted prices
-  Forecasts next day’s closing price
-  Generates dynamic plots with Streamlit
-  Accepts any valid Yahoo Finance ticker (e.g., `AAPL`, `TSLA`, `INFY.NS`)

---

##  Tech Stack

- **Frontend/UI:** Streamlit
- **Backend/ML:** Python, TensorFlow, Keras
- **Data Handling:** yFinance, NumPy, Pandas
- **Visualization:** Matplotlib
- **Preprocessing:** MinMaxScaler from scikit-learn

---

1 Clone the repository
```
https://github.com/Bedantaroy9/stock_price_predictor.git
cd stock_price_predictor
```
2 Install dependencies
```
pip install -r requirements.txt
```
3 Run the Streamlit app
```
streamlit run app.py
```

## Live App
https://bedantaroy-stock-price-prediction.hf.space/
