import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd

class ImprovedLSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=96, num_layers=2, output_size=1, dropout=0.3):
        super().__init__()
        # Main LSTM layer
        self.lstm1 = nn.LSTM(
            input_size, 
            hidden_size, 
            num_layers=1, 
            batch_first=True,
            bidirectional=True  # Bidirectional captures patterns from both directions
        )
        
        # Second LSTM layer with skip connection capability
        self.lstm2 = nn.LSTM(
            hidden_size*2,  # *2 because of bidirectional
            hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=False
        )
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size//2),
            nn.Tanh(),
            nn.Linear(hidden_size//2, 1),
            nn.Softmax(dim=1)
        )
        
        # Fully connected layers with batch normalization
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.leaky_relu = nn.LeakyReLU(0.1)  # LeakyReLU instead of ReLU
        self.dropout = nn.Dropout(dropout)
        
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size//2)
        
        self.fc3 = nn.Linear(hidden_size//2, output_size)
        
        # Initialize weights
        self.init_weights()
    
    def init_weights(self):
        """Initialize weights to help with training"""
        # LSTM weights
        for lstm in [self.lstm1, self.lstm2]:
            for name, param in lstm.named_parameters():
                if 'weight_ih' in name:
                    nn.init.xavier_uniform_(param.data)
                elif 'weight_hh' in name:
                    nn.init.orthogonal_(param.data)
                elif 'bias' in name:
                    param.data.fill_(0)
        
        # Attention weights            
        for layer in self.attention:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
                
        # FC layers
        for fc in [self.fc1, self.fc2, self.fc3]:
            nn.init.xavier_uniform_(fc.weight)
            nn.init.zeros_(fc.bias)

    def forward(self, x):
        # x shape: [batch_size, seq_len, input_size]
        batch_size, seq_len, _ = x.size()
        
        # First LSTM layer
        lstm1_out, _ = self.lstm1(x)  # [batch_size, seq_len, hidden_size*2]
        
        # Second LSTM layer
        lstm2_out, _ = self.lstm2(lstm1_out)  # [batch_size, seq_len, hidden_size]
        
        # Apply attention to focus on important time steps
        attn_weights = self.attention(lstm2_out)  # [batch_size, seq_len, 1]
        context = torch.bmm(attn_weights.transpose(1, 2), lstm2_out)  # [batch_size, 1, hidden_size]
        context = context.squeeze(1)  # [batch_size, hidden_size]
        
        # Fully connected layers with batch normalization
        out = self.bn1(context)
        out = self.fc1(out)
        out = self.leaky_relu(out)
        out = self.dropout(out)
        
        out = self.bn2(out)
        out = self.fc2(out)
        out = self.leaky_relu(out)
        out = self.dropout(out)
        
        out = self.fc3(out)
        
        return out

def prepare_lstm_data(data, sequence_length=60, target_col='Close', feature_cols=None, x_scaler=None, y_scaler=None):
    """
    Prepare data for LSTM model with multiple features and proper scaling
    
    Args:
        data: DataFrame with stock data
        sequence_length: Number of time steps to look back
        target_col: Column to predict
        feature_cols: List of feature columns to use (if None, use all numeric columns)
        x_scaler: Pre-fitted feature scaler (if None, a new one will be created)
        y_scaler: Pre-fitted target scaler (if None, a new one will be created)
    
    Returns:
        x_tensor, y_tensor, x_scaler, y_scaler
    """
    # If feature_cols is None, use all numeric columns
    if feature_cols is None:
        feature_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        # Ensure target_col is in feature_cols
        if target_col not in feature_cols:
            feature_cols.append(target_col)
    
    # Add technical indicators
    data = add_technical_indicators(data, target_col)
    
    # Create or use scalers
    if x_scaler is None:
        # Use StandardScaler for features (better for neural networks)
        x_scaler = StandardScaler()
        fit_x = True
    else:
        fit_x = False
        
    if y_scaler is None:
        # Use MinMaxScaler for target (keeps values in reasonable range)
        y_scaler = MinMaxScaler()
        fit_y = True
    else:
        fit_y = False
    
    # Scale features
    feature_data = data[feature_cols].copy()
    if fit_x:
        scaled_features = x_scaler.fit_transform(feature_data)
    else:
        scaled_features = x_scaler.transform(feature_data)
    
    scaled_features_df = pd.DataFrame(scaled_features, columns=feature_cols, index=data.index)
    
    # Scale target separately
    target_data = data[[target_col]].copy()
    if fit_y:
        scaled_target = y_scaler.fit_transform(target_data)
    else:
        scaled_target = y_scaler.transform(target_data)
    
    # Create sequences
    x, y = [], []
    for i in range(sequence_length, len(scaled_features)):
        # Use all features for input sequence
        x.append(scaled_features[i-sequence_length:i])
        # Use only target column for output
        y.append(scaled_target[i])
    
    # Convert to tensors
    x_tensor = torch.tensor(np.array(x), dtype=torch.float32)
    y_tensor = torch.tensor(np.array(y), dtype=torch.float32)
    
    return x_tensor, y_tensor, x_scaler, y_scaler

def add_technical_indicators(data, price_col='Close'):
    """Add basic technical indicators to the dataframe (simplified for speed)"""
    df = data.copy()
    
    # Moving averages (just a few key ones)
    df['MA5'] = df[price_col].rolling(window=5).mean()
    df['MA20'] = df[price_col].rolling(window=20).mean()
    
    # Exponential moving averages
    df['EMA12'] = df[price_col].ewm(span=12, adjust=False).mean()
    df['EMA26'] = df[price_col].ewm(span=26, adjust=False).mean()
    
    # MACD
    df['MACD'] = df['EMA12'] - df['EMA26']
    
    # Relative Strength Index (RSI)
    delta = df[price_col].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Simple price changes
    df['Returns'] = df[price_col].pct_change()
    df['Log_Returns'] = np.log(df[price_col] / df[price_col].shift(1))
    
    # Fill NaN values
    df = df.fillna(method='bfill').fillna(method='ffill')
    
    return df

def train_lstm(data, epochs=20, lr=0.0005, batch_size=64, sequence_length=30, 
             target_col='Close', feature_cols=None, val_split=0.15,
             use_early_stopping=True, use_scheduler=True, weight_decay=1e-5):
    """
    Train an enhanced LSTM model with advanced training techniques
    
    Args:
        data: DataFrame with stock data
        epochs: Number of training epochs
        lr: Initial learning rate
        batch_size: Batch size for training
        sequence_length: Number of time steps to look back
        target_col: Column to predict
        feature_cols: List of feature columns to use (if None, use all numeric columns)
        val_split: Validation split ratio
        use_early_stopping: Whether to use early stopping
        use_scheduler: Whether to use learning rate scheduling
        weight_decay: L2 regularization parameter
    
    Returns:
        model, x_scaler, y_scaler
    """
    # Prepare data
    x, y, x_scaler, y_scaler = prepare_lstm_data(data, sequence_length, target_col, feature_cols)
    
    # Print feature information
    print(f"Number of features: {x.shape[2]}")
    
    # Split into train and validation sets
    train_size = int(len(x) * (1 - val_split))
    x_train, x_val = x[:train_size], x[train_size:]
    y_train, y_val = y[:train_size], y[train_size:]
    
    print(f"Training samples: {len(x_train)}, Validation samples: {len(x_val)}")
    
    # Create DataLoader for batching
    train_dataset = TensorDataset(x_train, y_train)
    val_dataset = TensorDataset(x_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Initialize model
    input_size = x.shape[2]  # Number of features
    model = ImprovedLSTMModel(input_size=input_size)
    
    # Loss function and optimizer
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Learning rate scheduler
    if use_scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3, verbose=True
        )
    
    # Early stopping parameters
    if use_early_stopping:
        best_val_loss = float('inf')
        patience = 7  # Increased patience for better convergence
        patience_counter = 0
        best_model_state = None
    
    # Training loop
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = loss_fn(outputs, batch_y)
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                outputs = model(batch_x)
                loss = loss_fn(outputs, batch_y)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        
        # Learning rate scheduling
        if use_scheduler:
            scheduler.step(val_loss)
        
        # Print progress
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        
        # Early stopping
        if use_early_stopping:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
    
    # Load best model
    if use_early_stopping and best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"Loaded best model with validation loss: {best_val_loss:.6f}")
    
    return model, x_scaler, y_scaler
