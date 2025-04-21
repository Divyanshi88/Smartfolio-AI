import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import MinMaxScaler

class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=2, output_size=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

def prepare_lstm_data(data, sequence_length=50):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data[['Close']])
    x, y = [], []

    for i in range(sequence_length, len(scaled)):
        x.append(scaled[i-sequence_length:i])
        y.append(scaled[i])

    return torch.tensor(np.array(x), dtype=torch.float32), \
           torch.tensor(np.array(y), dtype=torch.float32), scaler

def train_lstm(data, epochs=50, lr=0.001):
    model = LSTMModel()
    x, y, scaler = prepare_lstm_data(data)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        output = model(x)
        loss = loss_fn(output, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Loss={loss.item()}")

    return model, scaler
