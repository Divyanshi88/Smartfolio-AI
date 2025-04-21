import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, GRU, Bidirectional
from keras.callbacks import EarlyStopping, ModelCheckpoint
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import warnings
warnings.filterwarnings('ignore')

# Data loading and preprocessing functions
def load_stock_data(file_path):
    """Load stock data from CSV file"""
    df = pd.read_csv(file_path)
    # Check if date column exists and convert to datetime
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
    return df

def preprocess_data(df):
    """Basic preprocessing for stock data"""
    # Check for missing values
    if df.isnull().sum().sum() > 0:
        print(f"Found {df.isnull().sum().sum()} missing values. Filling with forward fill method.")
        df = df.fillna(method='ffill')
        # If there are still NaN values at the beginning, use backward fill
        df = df.fillna(method='bfill')
    
    # Calculate additional features
    df['Returns'] = df['Adj Close'].pct_change()
    df['MA5'] = df['Adj Close'].rolling(window=5).mean()
    df['MA20'] = df['Adj Close'].rolling(window=20).mean()
    df['Volatility'] = df['Returns'].rolling(window=20).std()
    
    # Calculate MACD
    df['EMA12'] = df['Adj Close'].ewm(span=12, adjust=False).mean()
    df['EMA26'] = df['Adj Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA12'] - df['EMA26']
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    # Drop rows with NaN (due to rolling calculations)
    df = df.dropna()
    
    return df

def create_sequences(data, seq_length):
    """Create input sequences and targets for time series prediction"""
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:i+seq_length]
        y = data[i+seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

# ----------- LSTM Model -----------
def build_lstm_model(input_shape, units=50, dropout_rate=0.2):
    """Build LSTM model for time series prediction"""
    model = Sequential()
    model.add(LSTM(units=units, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(dropout_rate))
    model.add(LSTM(units=units))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1))
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def train_lstm_model(X_train, y_train, X_val, y_val, input_shape, epochs=50, batch_size=32):
    """Train LSTM model with early stopping"""
    model = build_lstm_model(input_shape)
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
    
    model_checkpoint = ModelCheckpoint(
        'lstm_model.h5',
        monitor='val_loss',
        save_best_only=True
    )
    
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping, model_checkpoint],
        verbose=1
    )
    
    return model, history

# ----------- BiLSTM Model -----------
def build_bilstm_model(input_shape, units=50, dropout_rate=0.2):
    """Build Bidirectional LSTM model"""
    model = Sequential()
    model.add(Bidirectional(LSTM(units=units, return_sequences=True), input_shape=input_shape))
    model.add(Dropout(dropout_rate))
    model.add(Bidirectional(LSTM(units=units)))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1))
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def train_bilstm_model(X_train, y_train, X_val, y_val, input_shape, epochs=50, batch_size=32):
    """Train Bidirectional LSTM model"""
    model = build_bilstm_model(input_shape)
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
    
    model_checkpoint = ModelCheckpoint(
        'bilstm_model.h5',
        monitor='val_loss',
        save_best_only=True
    )
    
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping, model_checkpoint],
        verbose=1
    )
    
    return model, history

# ----------- GRU Model -----------
def build_gru_model(input_shape, units=50, dropout_rate=0.2):
    """Build GRU model"""
    model = Sequential()
    model.add(GRU(units=units, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(dropout_rate))
    model.add(GRU(units=units))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1))
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def train_gru_model(X_train, y_train, X_val, y_val, input_shape, epochs=50, batch_size=32):
    """Train GRU model"""
    model = build_gru_model(input_shape)
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
    
    model_checkpoint = ModelCheckpoint(
        'gru_model.h5',
        monitor='val_loss',
        save_best_only=True
    )
    
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping, model_checkpoint],
        verbose=1
    )
    
    return model, history

# ----------- Transformer Model (PyTorch) -----------
class TimeSeriesTransformerDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x

class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_encoder_layers, dim_feedforward, dropout=0.1):
        super(TimeSeriesTransformer, self).__init__()
        self.input_projection = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layers = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, 
                                                   dim_feedforward=dim_feedforward, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_encoder_layers)
        self.output_projection = nn.Linear(d_model, 1)
        self.d_model = d_model

    def forward(self, src):
        # src shape: [batch_size, seq_len, input_dim]
        src = src.permute(1, 0, 2)  # [seq_len, batch_size, input_dim]
        src = self.input_projection(src) * np.sqrt(self.d_model)  # [seq_len, batch_size, d_model]
        src = self.pos_encoder(src)  # [seq_len, batch_size, d_model]
        output = self.transformer_encoder(src)  # [seq_len, batch_size, d_model]
        output = output[-1]  # Take the last sequence step [batch_size, d_model]
        output = self.output_projection(output)  # [batch_size, 1]
        return output

def train_transformer_model(X_train, y_train, X_val, y_val, input_dim, 
                           d_model=64, nhead=8, num_encoder_layers=4, 
                           dim_feedforward=256, epochs=50, batch_size=32):
    """Train Transformer model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create datasets and dataloaders
    train_dataset = TimeSeriesTransformerDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataset = TimeSeriesTransformerDataset(X_val, y_val)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Initialize model
    model = TimeSeriesTransformer(
        input_dim=input_dim,
        d_model=d_model,
        nhead=nhead,
        num_encoder_layers=num_encoder_layers,
        dim_feedforward=dim_feedforward
    ).to(device)
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # Training loop
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            output = model(batch_X)
            loss = criterion(output.squeeze(), batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                output = model(batch_X)
                val_loss += criterion(output.squeeze(), batch_y).item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        # Learning rate scheduler
        scheduler.step(val_loss)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'transformer_model.pth')
        
        print(f'Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
    
    # Load best model
    model.load_state_dict(torch.load('transformer_model.pth'))
    return model, {"train_losses": train_losses, "val_losses": val_losses}

# ----------- Informer Model (Based on Transformer with Improvements) -----------
class InformerEncoder(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_encoder_layers, dim_feedforward, dropout=0.1):
        super(InformerEncoder, self).__init__()
        self.input_projection = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Using standard transformer encoder layers but with modifications for Informer architecture
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_encoder_layers)
        self.d_model = d_model
    
    def forward(self, src):
        # src shape: [seq_len, batch_size, input_dim]
        src = self.input_projection(src) * np.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        return output

class InformerDecoder(nn.Module):
    def __init__(self, d_model, nhead, num_decoder_layers, dim_feedforward, dropout=0.1):
        super(InformerDecoder, self).__init__()
        decoder_layers = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layers, num_decoder_layers)
        self.output_projection = nn.Linear(d_model, 1)
    
    def forward(self, tgt, memory):
        # tgt shape: [seq_len, batch_size, d_model]
        # memory shape: [seq_len, batch_size, d_model]
        output = self.transformer_decoder(tgt, memory)
        output = output[-1]  # Take last sequence step
        output = self.output_projection(output)  # [batch_size, 1]
        return output

class InformerModel(nn.Module):
    def __init__(self, input_dim, d_model=64, nhead=8, num_encoder_layers=4, 
                 num_decoder_layers=4, dim_feedforward=256, dropout=0.1):
        super(InformerModel, self).__init__()
        self.encoder = InformerEncoder(
            input_dim=input_dim,
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        self.decoder = InformerDecoder(
            d_model=d_model,
            nhead=nhead,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        self.d_model = d_model
        
    def forward(self, src):
        # src shape: [batch_size, seq_len, input_dim]
        src = src.permute(1, 0, 2)  # [seq_len, batch_size, input_dim]
        
        # Create decoder input (last element of encoder sequence)
        memory = self.encoder(src)
        # Use last position for the decoder
        tgt = memory[-1:, :, :].clone()
        
        output = self.decoder(tgt, memory)
        return output

def train_informer_model(X_train, y_train, X_val, y_val, input_dim, 
                         d_model=64, nhead=8, num_encoder_layers=4, num_decoder_layers=4,
                         dim_feedforward=256, epochs=50, batch_size=32):
    """Train Informer model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create datasets and dataloaders
    train_dataset = TimeSeriesTransformerDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataset = TimeSeriesTransformerDataset(X_val, y_val)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Initialize model
    model = InformerModel(
        input_dim=input_dim,
        d_model=d_model,
        nhead=nhead,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        dim_feedforward=dim_feedforward
    ).to(device)
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # Training loop
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            output = model(batch_X)
            loss = criterion(output.squeeze(), batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                output = model(batch_X)
                val_loss += criterion(output.squeeze(), batch_y).item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        # Learning rate scheduler
        scheduler.step(val_loss)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'informer_model.pth')
        
        print(f'Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
    
    # Load best model
    model.load_state_dict(torch.load('informer_model.pth'))
    return model, {"train_losses": train_losses, "val_losses": val_losses}

# Main function to train all models
def train_all_models(file_path, target_col='Adj Close', sequence_length=60, test_size=0.2):
    """Train all models on the stock data"""
    # Load and preprocess data
    df = load_stock_data(file_path)
    df = preprocess_data(df)
    
    # Select features based on what's available in the dataframe
    feature_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Returns', 'MA5', 'MA20', 'Volatility', 'MACD', 'Signal']
    available_features = [col for col in feature_cols if col in df.columns]
    
    # Scale data
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    
    # Prepare feature matrix
    X_data = scaler_X.fit_transform(df[available_features])
    y_data = scaler_y.fit_transform(df[[target_col]])
    
    # Create sequences
    X, y = create_sequences(X_data, sequence_length)
    y = y[:, 0]  # Flatten y
    
    # Split data
    split_idx = int(len(X) * (1 - test_size))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Further split test data into validation and test
    val_idx = int(len(X_test) * 0.5)
    X_val, X_test = X_test[:val_idx], X_test[val_idx:]
    y_val, y_test = y_test[:val_idx], y_test[val_idx:]
    
    # Print shapes
    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")
    print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
    
    # Train models
    models = {}
    histories = {}
    
    # 1. LSTM
    print("\nTraining LSTM model...")
    input_shape = (X_train.shape[1], X_train.shape[2])
    lstm_model, lstm_history = train_lstm_model(X_train, y_train, X_val, y_val, input_shape)
    models['lstm'] = lstm_model
    histories['lstm'] = lstm_history
    
    # 2. BiLSTM
    print("\nTraining BiLSTM model...")
    bilstm_model, bilstm_history = train_bilstm_model(X_train, y_train, X_val, y_val, input_shape)
    models['bilstm'] = bilstm_model
    histories['bilstm'] = bilstm_history
    
    # 3. GRU
    print("\nTraining GRU model...")
    gru_model, gru_history = train_gru_model(X_train, y_train, X_val, y_val, input_shape)
    models['gru'] = gru_model
    histories['gru'] = gru_history
    
    # 4. Transformer
    print("\nTraining Transformer model...")
    transformer_model, transformer_history = train_transformer_model(
        X_train, y_train, X_val, y_val, input_dim=X_train.shape[2]
    )
    models['transformer'] = transformer_model
    histories['transformer'] = transformer_history
    
    # 5. Informer
    print("\nTraining Informer model...")
    informer_model, informer_history = train_informer_model(
        X_train, y_train, X_val, y_val, input_dim=X_train.shape[2]
    )
    models['informer'] = informer_model
    histories['informer'] = informer_history
    
    # Evaluate models
    evaluate_models(models, X_test, y_test, scaler_y, target_col)
    
    return models, histories, scaler_X, scaler_y

def evaluate_models(models, X_test, y_test, scaler_y, target_col):
    """Evaluate all models on test data"""
    print("\nModel Evaluation Results:")
    print("-" * 50)
    
    results = {}
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Prepare true values
    y_true = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()
    
    for name, model in models.items():
        if name in ['lstm', 'bilstm', 'gru']:
            # Tensorflow models
            y_pred = model.predict(X_test)
            y_pred = scaler_y.inverse_transform(y_pred).flatten()
            save_model_results("LSTM", y_test, y_pred)  # or "BiLSTM" / "GRU"
        else:
            # PyTorch models
            model.eval()
            with torch.no_grad():
                X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
                y_pred = model(X_test_tensor).squeeze().cpu().numpy()
            save_model_results("Transformer", y_test, y_pred)


        
        # Calculate metrics
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        results[name] = {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2
        }
        
        print(f"Model: {name}")
        print(f"MSE: {mse:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE: {mae:.4f}")
        print(f"R2: {r2:.4f}")
        print("-" * 50)
    
    return results

import csv

def save_model_results(model_name, y_true, y_pred, results_file='model_results.csv'):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    # Prepare the row
    row = {
        'Model': model_name,
        'RMSE': rmse,
        'MAE': mae,
        'R2_Score': r2
    }

    # Append to CSV
    try:
        with open(results_file, mode='x', newline='') as f:  # Create file with headers
            writer = csv.DictWriter(f, fieldnames=row.keys())
            writer.writeheader()
            writer.writerow(row)
    except FileExistsError:
        with open(results_file, mode='a', newline='') as f:  # Append if already exists
            writer = csv.DictWriter(f, fieldnames=row.keys())
            writer.writerow(row)

# Example usage
if __name__ == "__main__":
    # Example file path
    file_path = "D:/time series lab/project/project/stocks/A.csv"
    
    # Train all models
    models, histories, scaler_X, scaler_y = train_all_models(
        file_path=file_path,
        target_col='Adj Close',
        sequence_length=60,
        test_size=0.2
    )
    save_model_results()
    print("Model training completed!")