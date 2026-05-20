import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import pickle

# ---------------------------------------------------------------
# 1. DATASET CLASS
# ---------------------------------------------------------------
class TTFDataset(Dataset):
    def __init__(self, sequences, labels):
        self.X = torch.tensor(sequences, dtype=torch.float32)
        self.y = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ---------------------------------------------------------------
# 2. LSTM MODEL
# ---------------------------------------------------------------
class TTFPredictor(nn.Module):
    def __init__(self, input_size=5, hidden_size=64, num_layers=2, dropout=0.2):
        super(TTFPredictor, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]  # take last timestep output
        return self.fc(last_hidden).squeeze(-1)


# ---------------------------------------------------------------
# 3. LOAD & PREPARE DATA
# ---------------------------------------------------------------
def load_data(csv_path='data/lstm_training_data.csv', sequence_length=20):
    print("Loading data...")
    df = pd.read_csv(csv_path)

    # Extract feature columns (t0 to t19, 5 sensors each)
    feature_cols = []
    for t in range(sequence_length):
        feature_cols.extend([
            f't{t}_temp', f't{t}_vib', f't{t}_wear',
            f't{t}_health', f't{t}_hsm'
        ])

    X_flat = df[feature_cols].values  # shape: (N, 100)
    y = df['ttf'].values              # shape: (N,)

    # Reshape X into sequences: (N, sequence_length, n_features)
    N = len(X_flat)
    X = X_flat.reshape(N, sequence_length, 5)

    # Normalize features using StandardScaler on flattened then reshape back
    scaler = StandardScaler()
    X_flat_scaled = scaler.fit_transform(X_flat)
    X_scaled = X_flat_scaled.reshape(N, sequence_length, 5)

    # Normalize TTF to 0-1 range for stable training
    ttf_max = y.max()
    y_normalized = y / ttf_max

    print(f"X shape: {X_scaled.shape}")
    print(f"y shape: {y_normalized.shape}")
    print(f"TTF max (for denormalization): {ttf_max}")

    return X_scaled, y_normalized, scaler, ttf_max


# ---------------------------------------------------------------
# 4. TRAINING LOOP
# ---------------------------------------------------------------
def train(csv_path='data/lstm_training_data.csv',
          model_save_path='models/lstm_ttf.pth',
          scaler_save_path='models/lstm_scaler.pkl'):

    os.makedirs('models', exist_ok=True)

    # Load data
    X, y, scaler, ttf_max = load_data(csv_path)

    # Train/val split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    train_dataset = TTFDataset(X_train, y_train)
    val_dataset   = TTFDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=256, shuffle=False)

    # Model, loss, optimizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nTraining on: {device}")

    model = TTFPredictor(input_size=5, hidden_size=64, num_layers=2, dropout=0.2)
    model.to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=3, factor=0.5)

    # Training
    best_val_loss = float('inf')
    epochs = 30

    print(f"\nStarting training for {epochs} epochs...\n")

    for epoch in range(epochs):
        # --- Train ---
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # --- Validate ---
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                preds = model(X_batch)
                loss = criterion(preds, y_batch)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        scheduler.step(val_loss)

        # Convert loss back to hours for readability
        train_mae_hours = (train_loss ** 0.5) * ttf_max
        val_mae_hours   = (val_loss ** 0.5) * ttf_max

        print(f"Epoch {epoch+1:02d}/{epochs} | "
              f"Train Loss: {train_loss:.5f} (~{train_mae_hours:.1f}h) | "
              f"Val Loss: {val_loss:.5f} (~{val_mae_hours:.1f}h)")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_save_path)
            print(f"  -> Best model saved")

    # Save scaler and ttf_max for inference
    with open(scaler_save_path, 'wb') as f:
        pickle.dump({'scaler': scaler, 'ttf_max': ttf_max}, f)

    print(f"\nTraining complete.")
    print(f"Best val loss: ~{(best_val_loss**0.5)*ttf_max:.1f} hours error")
    print(f"Model saved to: {model_save_path}")
    print(f"Scaler saved to: {scaler_save_path}")


if __name__ == '__main__':
    train()