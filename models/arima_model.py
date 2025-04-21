from statsmodels.tsa.arima.model import ARIMA

def train_arima(data, order=(5, 1, 0)):
    model = ARIMA(data['Close'], order=order)
    model_fit = model.fit()
    return model_fit
