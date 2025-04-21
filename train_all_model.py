from utils.data_loader import load_stock_data
from models.arima_model import train_arima
from models.lstm_model import train_lstm
from models.tft_model import train_tft
# from models.informer_model import train_informer  # Optional

data = load_stock_data("your_stock_data.csv")

# ARIMA
arima_model = train_arima(data)

# LSTM
lstm_model, lstm_scaler = train_lstm(data)

# TFT
tft_model = train_tft(data)

# Informer (optional)
# informer_model = train_informer(data)

