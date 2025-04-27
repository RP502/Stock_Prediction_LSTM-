# stock_forecast_lstm_forecast.py
import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.callbacks import EarlyStopping
from datetime import timedelta

# Ignore warnings
warnings.filterwarnings('ignore')

# Streamlit app
st.title("📈 Dự đoán giá cổ phiếu bằng LSTM + Dự đoán tương lai")

# User input
ticker = st.text_input("Nhập mã cổ phiếu (ví dụ: MSFT, AAPL, VNM, VIC):", "MSFT")
start_date = st.date_input("Ngày bắt đầu", pd.to_datetime("2010-01-01"))
end_date = st.date_input("Ngày kết thúc", pd.to_datetime("2025-04-25"))

# Thêm tùy chọn
window_size = st.selectbox("Chọn số ngày dùng để dự đoán (window size)", [30, 60, 90], index=1)
epochs = st.slider("Số epochs để huấn luyện:", min_value=10, max_value=100, step=5, value=25)
future_days = st.selectbox("Dự đoán thêm bao nhiêu ngày trong tương lai?", [7, 30], index=0)

if st.button("Tải dữ liệu và Huấn luyện"):
    # 1. Tải dữ liệu
    df = yf.download(ticker, start=start_date, end=end_date, auto_adjust=False)[['Adj Close', 'Open', 'High', 'Low', 'Close', 'Volume']]
    df = df.round(2)

    st.subheader("Dữ liệu giá cổ phiếu")
    st.dataframe(df.tail())

    # 2. Vẽ biểu đồ giá
    st.subheader("Biểu đồ giá đóng cửa điều chỉnh")
    fig, ax = plt.subplots(figsize=(15, 7))
    df['Adj Close'].plot(ax=ax, grid=True, title=f"Giá đóng cửa điều chỉnh của {ticker}")
    st.pyplot(fig)

    # 3. Chuẩn bị dữ liệu
    adj_close = df[['Adj Close']]
    data = adj_close.values

    # Split train-test
    split = int(0.8 * len(data))
    train, test = data[:split], data[split:]

    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(data)

    # Tạo bộ dữ liệu cho LSTM
    X_train, y_train = [], []
    for i in range(window_size, len(train)):
        X_train.append(scaled_data[i-window_size:i, 0])
        y_train.append(scaled_data[i, 0])

    X_train, y_train = np.array(X_train), np.array(y_train)

    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    # 4. Xây dựng mô hình
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(1))

    model.compile(loss='mean_squared_error', optimizer='adam')

    # Early stopping
    early_stop = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)

    # 5. Train model
    with st.spinner('Đang huấn luyện mô hình...'):
        model.fit(X_train, y_train, epochs=epochs, batch_size=32, verbose=0, callbacks=[early_stop])

    st.success('Huấn luyện xong!')

    # 6. Dự đoán trên tập test
    inputs = data[len(data) - len(test) - window_size:]
    inputs = inputs.reshape(-1,1)
    inputs = scaler.transform(inputs)

    X_test = []
    for i in range(window_size, inputs.shape[0]):
        X_test.append(inputs[i-window_size:i,0])

    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1],1))

    predicted_prices = model.predict(X_test)
    predicted_prices = scaler.inverse_transform(predicted_prices)

    # Tính RMSE
    rmse = np.sqrt(np.mean(np.square(test - predicted_prices)))
    st.write(f"**RMSE của mô hình:** {rmse:.2f}")

    # 7. Hiển thị kết quả
    train_df = adj_close[:split]
    test_df = adj_close[split:]
    test_df['Predictions'] = predicted_prices

    st.subheader("Biểu đồ so sánh giá thực tế và dự đoán")
    fig2, ax2 = plt.subplots(figsize=(20,10))
    sns.set_style("whitegrid")
    ax2.plot(train_df['Adj Close'], label='Training')
    ax2.plot(test_df['Adj Close'], label='Actual')
    ax2.plot(test_df['Predictions'], label='Predicted')
    ax2.set_title(f"Giá cổ phiếu {ticker} - Thực tế vs Dự đoán", fontsize=20)
    ax2.set_xlabel('Ngày', fontsize=15)
    ax2.set_ylabel('Giá', fontsize=15)
    ax2.legend()
    st.pyplot(fig2)

    # 8. Dự đoán tương lai
    st.subheader(f"Dự đoán giá {future_days} ngày tới")

    last_inputs = scaled_data[-window_size:].tolist()
    future_predictions = []

    for _ in range(future_days):
        current_input = np.array(last_inputs[-window_size:]).reshape(1, window_size, 1)
        next_price = model.predict(current_input, verbose=0)
        future_predictions.append(next_price[0,0])
        last_inputs.append([next_price[0,0]])

    future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1,1))

    # Tạo dataframe cho dự đoán tương lai
    last_date = df.index[-1]
    future_dates = [last_date + timedelta(days=i) for i in range(1, future_days+1)]

    future_df = pd.DataFrame({'Date': future_dates, 'Predicted Price': future_predictions.flatten()})
    st.dataframe(future_df)

    # Vẽ dự đoán tương lai
    fig3, ax3 = plt.subplots(figsize=(20,10))
    ax3.plot(df['Adj Close'], label='Lịch sử')
    ax3.plot(future_df['Date'], future_df['Predicted Price'], label='Dự đoán tương lai', linestyle='dashed')
    ax3.set_title(f"Dự đoán giá cổ phiếu {ticker} {future_days} ngày tới", fontsize=20)
    ax3.set_xlabel('Năm', fontsize=15)
    ax3.set_ylabel('Giá', fontsize=15)
    ax3.legend()
    st.pyplot(fig3)

    # 9. Cho phép tải file CSV
    st.subheader("Tải kết quả dự đoán")

    # Gộp cả dự đoán test + future
    all_results = pd.concat([
        test_df.reset_index()[['Date', 'Adj Close', 'Predictions']],
        future_df.rename(columns={'Predicted Price': 'Predictions'}).assign(**{'Adj Close': np.nan})
    ], ignore_index=True)

    csv = all_results.to_csv(index=False).encode('utf-8')

    st.download_button(
        label="📄 Tải file CSV kết quả",
        data=csv,
        file_name=f'{ticker}_full_prediction.csv',
        mime='text/csv',
    )
