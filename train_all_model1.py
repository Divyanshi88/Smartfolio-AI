import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import csv
from sklearn.preprocessing import MinMaxScaler, StandardScaler
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
    """Enhanced preprocessing for stock data with advanced technical indicators"""
    # Check for missing values
    if df.isnull().sum().sum() > 0:
        print(f"Found {df.isnull().sum().sum()} missing values. Filling with forward fill method.")
        df = df.fillna(method='ffill')
        # If there are still NaN values at the beginning, use backward fill
        df = df.fillna(method='bfill')
    
    # Make sure we have the required columns
    required_cols = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
    for col in required_cols:
        if col not in df.columns:
            if col == 'Adj Close' and 'Close' in df.columns:
                df['Adj Close'] = df['Close']
            else:
                raise ValueError(f"Required column {col} not found in dataframe")
    
    # Basic price features
    df['Returns'] = df['Adj Close'].pct_change()
    df['Log_Returns'] = np.log(df['Adj Close'] / df['Adj Close'].shift(1))
    
    # Moving averages
    for window in [5, 10, 20, 50, 100]:
        df[f'MA{window}'] = df['Adj Close'].rolling(window=window).mean()
        # Distance from moving average (normalized)
        df[f'Dist_MA{window}'] = (df['Adj Close'] - df[f'MA{window}']) / df[f'MA{window}']
    
    # Exponential moving averages
    for window in [5, 12, 26, 50]:
        df[f'EMA{window}'] = df['Adj Close'].ewm(span=window, adjust=False).mean()
    
    # MACD
    df['MACD'] = df['EMA12'] - df['EMA26']
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
    
    # Bollinger Bands
    for window in [20]:
        df[f'BB_Middle_{window}'] = df['Adj Close'].rolling(window=window).mean()
        df[f'BB_Std_{window}'] = df['Adj Close'].rolling(window=window).std()
        df[f'BB_Upper_{window}'] = df[f'BB_Middle_{window}'] + 2 * df[f'BB_Std_{window}']
        df[f'BB_Lower_{window}'] = df[f'BB_Middle_{window}'] - 2 * df[f'BB_Std_{window}']
        # Normalized position within Bollinger Bands (0 = lower band, 1 = upper band)
        df[f'BB_Position_{window}'] = (df['Adj Close'] - df[f'BB_Lower_{window}']) / (df[f'BB_Upper_{window}'] - df[f'BB_Lower_{window}'])
    
    # Relative Strength Index (RSI)
    delta = df['Adj Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Stochastic Oscillator
    n = 14
    df['Stoch_K'] = 100 * ((df['Adj Close'] - df['Low'].rolling(n).min()) / 
                          (df['High'].rolling(n).max() - df['Low'].rolling(n).min()))
    df['Stoch_D'] = df['Stoch_K'].rolling(3).mean()
    
    # Average True Range (ATR) - Volatility indicator
    df['TR'] = np.maximum(
        df['High'] - df['Low'],
        np.maximum(
            abs(df['High'] - df['Adj Close'].shift(1)),
            abs(df['Low'] - df['Adj Close'].shift(1))
        )
    )
    df['ATR'] = df['TR'].rolling(window=14).mean()
    
    # Average Directional Index (ADX) - Trend strength indicator
    df['Plus_DM'] = np.where(
        (df['High'] - df['High'].shift(1)) > (df['Low'].shift(1) - df['Low']),
        np.maximum(df['High'] - df['High'].shift(1), 0),
        0
    )
    df['Minus_DM'] = np.where(
        (df['Low'].shift(1) - df['Low']) > (df['High'] - df['High'].shift(1)),
        np.maximum(df['Low'].shift(1) - df['Low'], 0),
        0
    )
    df['Plus_DI'] = 100 * (df['Plus_DM'].rolling(window=14).mean() / df['ATR'])
    df['Minus_DI'] = 100 * (df['Minus_DM'].rolling(window=14).mean() / df['ATR'])
    df['DX'] = 100 * abs(df['Plus_DI'] - df['Minus_DI']) / (df['Plus_DI'] + df['Minus_DI'])
    df['ADX'] = df['DX'].rolling(window=14).mean()
    
    # Volume features
    df['Volume_Change'] = df['Volume'].pct_change()
    df['Volume_MA10'] = df['Volume'].rolling(window=10).mean()
    df['Relative_Volume'] = df['Volume'] / df['Volume_MA10']
    
    # Price-volume relationship
    df['Price_Volume_Trend'] = (df['Adj Close'].pct_change() * df['Volume']).cumsum()
    
    # On-Balance Volume (OBV)
    df['OBV_Change'] = np.where(
        df['Adj Close'] > df['Adj Close'].shift(1),
        df['Volume'],
        np.where(
            df['Adj Close'] < df['Adj Close'].shift(1),
            -df['Volume'],
            0
        )
    )
    df['OBV'] = df['OBV_Change'].cumsum()
    
    # Momentum indicators
    for n in [3, 5, 10, 20]:
        # Rate of Change
        df[f'ROC_{n}'] = df['Adj Close'].pct_change(periods=n) * 100
        # Momentum
        df[f'Momentum_{n}'] = df['Adj Close'] - df['Adj Close'].shift(n)
    
    # Commodity Channel Index (CCI)
    df['TP'] = (df['High'] + df['Low'] + df['Adj Close']) / 3
    df['TP_MA20'] = df['TP'].rolling(window=20).mean()
    df['TP_Dev'] = abs(df['TP'] - df['TP_MA20'])
    df['TP_Dev_MA20'] = df['TP_Dev'].rolling(window=20).mean()
    df['CCI'] = (df['TP'] - df['TP_MA20']) / (0.015 * df['TP_Dev_MA20'])
    
    # Williams %R
    df['Williams_R'] = -100 * ((df['High'].rolling(14).max() - df['Adj Close']) / 
                             (df['High'].rolling(14).max() - df['Low'].rolling(14).min()))
    
    # Drop rows with NaN (due to rolling calculations)
    df = df.dropna()
    
    # Remove features with high correlation to avoid multicollinearity
    # This is a simple approach - in production, you might want to use a more sophisticated method
    corr_matrix = df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
    
    print(f"Dropping {len(to_drop)} highly correlated features: {to_drop}")
    df = df.drop(columns=to_drop)
    
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
def build_lstm_model(input_shape, units=128, dropout_rate=0.3):
    """Build improved LSTM model for time series prediction"""
    model = Sequential()
    
    # First LSTM layer with more units and return sequences
    model.add(LSTM(units=units, 
                  return_sequences=True, 
                  input_shape=input_shape,
                  recurrent_dropout=0.1))  # Adding recurrent dropout
    model.add(Dropout(dropout_rate))
    
    # Second LSTM layer with return sequences
    model.add(LSTM(units=units, 
                  return_sequences=True,
                  recurrent_dropout=0.1))
    model.add(Dropout(dropout_rate))
    
    # Third LSTM layer
    model.add(LSTM(units=units))
    model.add(Dropout(dropout_rate))
    
    # Adding more dense layers with batch normalization for better generalization
    model.add(Dense(units=64))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU(alpha=0.1))  # Using LeakyReLU instead of ReLU
    model.add(Dropout(dropout_rate/2))
    
    # Output layer
    model.add(Dense(1))
    
    # Using a more sophisticated optimizer with learning rate scheduling
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=0.001,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-07,
        amsgrad=True
    )
    
    # Compile with Huber loss which is less sensitive to outliers than MSE
    model.compile(
        optimizer=optimizer, 
        loss=tf.keras.losses.Huber(delta=1.0),  # Huber loss is more robust to outliers
        metrics=['mae', 'mse']  # Track multiple metrics
    )
    
    return model

def train_lstm_model(X_train, y_train, X_val, y_val, input_shape, epochs=100, batch_size=32):
    """Train LSTM model with enhanced training process"""
    model = build_lstm_model(input_shape)
    
    # More sophisticated early stopping
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True,
        min_delta=0.0001  # Minimum change to qualify as improvement
    )
    
    # Model checkpoint
    model_checkpoint = ModelCheckpoint(
        'lstm_model.h5',
        monitor='val_loss',
        save_best_only=True,
        save_weights_only=False
    )
    
    # Learning rate scheduler
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=0.00001,
        verbose=1
    )
    
    # Data augmentation for time series
    def time_series_augmentation(x, y, jitter_strength=0.1):
        # Add random noise to input sequences (jitter)
        x_aug = x + np.random.normal(0, jitter_strength, size=x.shape)
        return x_aug, y
    
    # Apply augmentation to a portion of the training data
    aug_size = int(len(X_train) * 0.3)  # Augment 30% of the data
    X_aug, y_aug = time_series_augmentation(X_train[:aug_size], y_train[:aug_size])
    
    # Combine original and augmented data
    X_train_combined = np.vstack([X_train, X_aug])
    y_train_combined = np.concatenate([y_train, y_aug])
    
    # Train with combined data
    history = model.fit(
        X_train_combined, y_train_combined,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping, model_checkpoint, reduce_lr],
        shuffle=True,  # Shuffle training data
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
    """Enhanced evaluation of all models on test data with additional metrics and visualizations"""
    print("\nModel Evaluation Results:")
    print("-" * 50)
    
    results = {}
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Prepare true values
    y_true = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()
    
    for name, model in models.items():
        print(f"\nEvaluating model: {name}")
        
        if name in ['lstm', 'bilstm', 'gru']:
            # Tensorflow models
            y_pred_scaled = model.predict(X_test)
            y_pred = scaler_y.inverse_transform(y_pred_scaled).flatten()
            
            # Save model results with the correct model name
            model_name_map = {
                'lstm': 'LSTM',
                'bilstm': 'BiLSTM',
                'gru': 'GRU'
            }
            save_model_results(model_name_map.get(name, name.upper()), y_true, y_pred)
        else:
            # PyTorch models
            model.eval()
            with torch.no_grad():
                X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
                y_pred_scaled = model(X_test_tensor).squeeze().cpu().numpy()
                
                # Reshape for inverse transform if needed
                if len(y_pred_scaled.shape) == 1:
                    y_pred_scaled = y_pred_scaled.reshape(-1, 1)
                
                y_pred = scaler_y.inverse_transform(y_pred_scaled).flatten()
            
            # Save model results
            save_model_results(name.capitalize(), y_true, y_pred)
        
        # Calculate standard metrics
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        # Calculate additional metrics
        mape = np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), 1e-10))) * 100
        
        # Direction accuracy (how often the model correctly predicts the direction of change)
        y_true_direction = np.diff(y_true) > 0
        y_pred_direction = np.diff(y_pred) > 0
        direction_accuracy = np.mean(y_true_direction == y_pred_direction) * 100
        
        # Store all metrics
        results[name] = {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2,
            'MAPE': mape,
            'Direction_Accuracy': direction_accuracy
        }
        
        # Print metrics
        print(f"MSE: {mse:.6f}")
        print(f"RMSE: {rmse:.6f}")
        print(f"MAE: {mae:.6f}")
        print(f"R2: {r2:.6f}")
        print(f"MAPE: {mape:.2f}%")
        print(f"Direction Accuracy: {direction_accuracy:.2f}%")
        
        # Plot predictions vs actual
        try:
            plt.figure(figsize=(12, 6))
            plt.plot(y_true, label='Actual')
            plt.plot(y_pred, label='Predicted')
            plt.title(f'{name.capitalize()} Model: Actual vs Predicted')
            plt.xlabel('Time Steps')
            plt.ylabel(target_col)
            plt.legend()
            plt.savefig(f'{name}_predictions.png')
            plt.close()
            print(f"Saved prediction plot to {name}_predictions.png")
            
            # Plot prediction error
            plt.figure(figsize=(12, 6))
            error = y_true - y_pred
            plt.plot(error)
            plt.title(f'{name.capitalize()} Model: Prediction Error')
            plt.xlabel('Time Steps')
            plt.ylabel('Error')
            plt.axhline(y=0, color='r', linestyle='-')
            plt.savefig(f'{name}_error.png')
            plt.close()
            print(f"Saved error plot to {name}_error.png")
            
            # Plot error distribution
            plt.figure(figsize=(10, 6))
            sns.histplot(error, kde=True)
            plt.title(f'{name.capitalize()} Model: Error Distribution')
            plt.xlabel('Error')
            plt.savefig(f'{name}_error_distribution.png')
            plt.close()
            print(f"Saved error distribution plot to {name}_error_distribution.png")
        except Exception as e:
            print(f"Error creating plots: {e}")
        
        print("-" * 50)
    
    # Compare all models
    try:
        metrics_df = pd.DataFrame({model: {metric: value for metric, value in model_results.items() 
                                         if metric in ['RMSE', 'MAE', 'R2', 'Direction_Accuracy']} 
                                 for model, model_results in results.items()})
        
        # Save comparison to CSV
        metrics_df.to_csv('model_comparison.csv')
        print("Saved model comparison to model_comparison.csv")
        
        # Create comparison plots
        for metric in ['RMSE', 'MAE', 'R2', 'Direction_Accuracy']:
            plt.figure(figsize=(10, 6))
            plt.bar(metrics_df.columns, metrics_df.loc[metric])
            plt.title(f'Model Comparison: {metric}')
            plt.ylabel(metric)
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(f'comparison_{metric}.png')
            plt.close()
            print(f"Saved comparison plot for {metric} to comparison_{metric}.png")
    except Exception as e:
        print(f"Error creating comparison: {e}")
    
    return results

import csv

def save_model_results(model_name, y_true, y_pred, results_file='model_results.csv'):
    """
    Save model evaluation results to CSV with enhanced metrics and error handling
    
    Args:
        model_name: Name of the model
        y_true: True values
        y_pred: Predicted values
        results_file: Path to CSV file for results
    """
    # Ensure arrays are the same length
    min_len = min(len(y_true), len(y_pred))
    y_true = y_true[:min_len]
    y_pred = y_pred[:min_len]
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), 1e-10))) * 100
    
    # Calculate direction accuracy if we have enough data points
    if len(y_true) > 1:
        y_true_direction = np.diff(y_true) > 0
        y_pred_direction = np.diff(y_pred) > 0
        direction_accuracy = np.mean(y_true_direction == y_pred_direction) * 100
    else:
        direction_accuracy = np.nan
    
    # Prepare the row with additional metrics
    row = {
        'Model': model_name,
        'RMSE': rmse,
        'MAE': mae,
        'R2_Score': r2,
        'MAPE': mape,
        'Direction_Accuracy': direction_accuracy
    }
    
    print(f"\nSaving results for {model_name}:")
    print(f"RMSE: {rmse:.6f}")
    print(f"MAE: {mae:.6f}")
    print(f"R2 Score: {r2:.6f}")
    print(f"MAPE: {mape:.2f}%")
    print(f"Direction Accuracy: {direction_accuracy:.2f}%")
    
    # Append to CSV with error handling
    try:
        # Check if file exists
        file_exists = os.path.isfile(results_file)
        
        with open(results_file, mode='a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            
            # Write header only if file is new
            if not file_exists:
                writer.writeheader()
            
            writer.writerow(row)
        
        print(f"Results saved to {results_file}")
    except Exception as e:
        print(f"Error saving results to CSV: {e}")
        
        # Fallback: print results to console
        print("Results (not saved to file):")
        for key, value in row.items():
            print(f"{key}: {value}")

# Example usage
if __name__ == "__main__":
    # Example file path - update this to your actual file path
    file_path = input("Enter the path to your stock data CSV file: ")
    
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        print("Using default example file path...")
        file_path = "stocks/A.csv"  # Default path
    
    # Get target column
    target_col = input("Enter the target column to predict (default: Adj Close): ") or "Adj Close"
    
    # Get sequence length
    try:
        sequence_length = int(input("Enter sequence length (default: 60): ") or "60")
    except ValueError:
        sequence_length = 60
        print("Invalid input, using default sequence length: 60")
    
    # Get test size
    try:
        test_size = float(input("Enter test size (0.0-1.0, default: 0.2): ") or "0.2")
        if test_size <= 0 or test_size >= 1:
            raise ValueError("Test size must be between 0 and 1")
    except ValueError:
        test_size = 0.2
        print("Invalid input, using default test size: 0.2")
    
    print("\n=== Starting Model Training ===")
    print(f"File: {file_path}")
    print(f"Target Column: {target_col}")
    print(f"Sequence Length: {sequence_length}")
    print(f"Test Size: {test_size}")
    print("=" * 30)
    
    # Train all models
    try:
        models, histories, scaler_X, scaler_y = train_all_models(
            file_path=file_path,
            target_col=target_col,
            sequence_length=sequence_length,
            test_size=test_size
        )
        print("\nModel training completed successfully!")
        
        # Save models
        print("\nSaving models...")
        
        # Save TensorFlow models
        for name in ['lstm', 'bilstm', 'gru']:
            if name in models:
                try:
                    models[name].save(f"{name}_model")
                    print(f"Saved {name} model to {name}_model directory")
                except Exception as e:
                    print(f"Error saving {name} model: {e}")
        
        # Save PyTorch models
        for name in ['transformer', 'informer']:
            if name in models:
                try:
                    torch.save(models[name].state_dict(), f"{name}_model.pth")
                    print(f"Saved {name} model to {name}_model.pth")
                except Exception as e:
                    print(f"Error saving {name} model: {e}")
        
        print("\nAll models trained and saved successfully!")
        
    except Exception as e:
        print(f"\nError during model training: {e}")
        import traceback
        traceback.print_exc()