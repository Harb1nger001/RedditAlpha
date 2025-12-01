import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
import joblib  # <-- for saving scaler

# -------------------------------------------------------
# CONFIG
# -------------------------------------------------------
SEQ_LEN = 30
BATCH_SIZE = 64
EPOCHS = 50
LR = 0.001
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TARGET_COL = "Close"

MODEL_SAVE_PATH = "Model/trained_arima_dl_model.pt"
SCALER_SAVE_PATH = "Model/scaler.save"

print("Using:", DEVICE)

# -------------------------------------------------------
# LOAD DATA
# -------------------------------------------------------
df = pd.read_csv("Data/Processed/merged_stock_sentiment_data.csv")

numeric_cols = [
    'Open', 'High', 'Low', 'Close', 'Volume',
    'SMA_14', 'EMA_14', 'RSI_14', 'MACD', 'MACD_signal', 'MACD_diff',
    'BB_Upper', 'BB_Lower', 'Momentum_10', 'OBV',
    'count_negative', 'count_positive', 'count_total',
    'positive_ratio', 'activity_count'
]

df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
df = df.dropna(subset=numeric_cols + [TARGET_COL]).reset_index(drop=True)

tickers = df["Ticker"].unique().tolist()
print(f"Found {len(tickers)} tickers.")

# -------------------------------------------------------
# SCALE FEATURES
# -------------------------------------------------------
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df[numeric_cols])

scaled_df = pd.DataFrame(scaled_data, columns=numeric_cols)
scaled_df["Ticker"] = df["Ticker"].values
scaled_df["Date"] = df["Date"].values

# Save the scaler
os.makedirs(os.path.dirname(SCALER_SAVE_PATH), exist_ok=True)
joblib.dump(scaler, SCALER_SAVE_PATH)
print(f"Scaler saved to {SCALER_SAVE_PATH}")

# -------------------------------------------------------
# DATASET CLASS
# -------------------------------------------------------
class MultiStockDataset(Dataset):
    def __init__(self, df, seq_len, target_col="Close"):
        self.seq_len = seq_len
        self.data = df
        self.target_col = target_col

        self.X = []
        self.Y = []
        self.T = []  # keep tickers for evaluation later

        values = df[numeric_cols].values
        tickers = df["Ticker"].values

        for i in range(len(df) - seq_len):
            seq_x = values[i:i+seq_len]
            seq_y = values[i+seq_len, numeric_cols.index(target_col)]
            self.X.append(seq_x.astype(np.float32))
            self.Y.append(float(seq_y))
            self.T.append(tickers[i+seq_len])

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.X[idx], dtype=torch.float32),
            torch.tensor(self.Y[idx], dtype=torch.float32),
            self.T[idx]
        )

# -------------------------------------------------------
# TRAIN/VAL/TEST SPLIT
# -------------------------------------------------------
dataset = MultiStockDataset(scaled_df, SEQ_LEN)

N = len(dataset)
train_n = int(0.80 * N)
val_n = int(0.10 * N)
test_n = N - train_n - val_n

train_set, val_set, test_set = torch.utils.data.random_split(
    dataset, [train_n, val_n, test_n]
)

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE)

print(f"Sequences -> train: {train_n}, val: {val_n}, test: {test_n}")

# -------------------------------------------------------
# ARIMA-STYLE MODEL (Deep)
# -------------------------------------------------------
class DeepARIMA(nn.Module):
    def __init__(self, input_size, hidden_size=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size * SEQ_LEN, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        self.backcast = nn.Linear(hidden_size, input_size)
        self.forecast = nn.Linear(hidden_size, 1)

    def forward(self, x):
        batch = x.size(0)
        x = x.reshape(batch, -1)
        h = self.encoder(x)
        _ = self.backcast(h)
        return self.forecast(h).squeeze(-1)

model = DeepARIMA(input_size=len(numeric_cols)).to(DEVICE)

# -------------------------------------------------------
# LOSS + OPT
# -------------------------------------------------------
criterion = nn.HuberLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

best_val_loss = float("inf")

# -------------------------------------------------------
# TRAINING LOOP
# -------------------------------------------------------
print("\n=== Training Model ===")

for epoch in range(1, EPOCHS+1):
    model.train()
    train_losses = []

    for x, y, _ in train_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        preds = model(x)
        loss = criterion(preds, y)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

    # Validation
    model.eval()
    val_losses = []
    with torch.no_grad():
        for x, y, _ in val_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            preds = model(x)
            loss = criterion(preds, y)
            val_losses.append(loss.item())

    avg_train = np.mean(train_losses)
    avg_val = np.mean(val_losses)
    scheduler.step(avg_val)

    print(f"Epoch {epoch}/{EPOCHS} — Train: {avg_train:.4f} — Val: {avg_val:.4f}")

    if avg_val < best_val_loss:
        best_val_loss = avg_val
        os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print("  ✔ Model improved → saved.")

print("\nTraining complete. Best val loss =", best_val_loss)


# -------------------------------------------------------
# LOAD BEST MODEL
# -------------------------------------------------------
model.load_state_dict(torch.load(MODEL_SAVE_PATH))
model.eval()

# -------------------------------------------------------
# EVALUATION (OVERALL + PER TICKER)
# -------------------------------------------------------
all_preds = []
all_true = []
all_tickers = []

with torch.no_grad():
    for x, y, t in test_loader:
        x = x.to(DEVICE)
        preds = model(x).cpu().numpy()
        all_preds.extend(preds)
        all_true.extend(y.numpy())
        all_tickers.extend(t)

# Overall metrics
rmse = np.sqrt(mean_squared_error(all_true, all_preds))
mae = mean_absolute_error(all_true, all_preds)
r2 = r2_score(all_true, all_preds)

print("\n=== OVERALL TEST METRICS ===")
print("RMSE :", rmse)
print("MAE  :", mae)
print("R²   :", r2)

# Per-ticker metrics
print("\n=== PER-TICKER METRICS ===")
ticker_metrics = {}

for ticker in tickers:
    idx = [i for i, t in enumerate(all_tickers) if t == ticker]
    if len(idx) == 0:
        continue

    y_true_t = [all_true[i] for i in idx]
    y_pred_t = [all_preds[i] for i in idx]

    rmse_t = np.sqrt(mean_squared_error(y_true_t, y_pred_t))
    mae_t = mean_absolute_error(y_true_t, y_pred_t)
    r2_t = r2_score(y_true_t, y_pred_t)

    ticker_metrics[ticker] = (rmse_t, mae_t, r2_t)
    print(f"{ticker} → RMSE={rmse_t:.4f}, MAE={mae_t:.4f}, R²={r2_t:.4f}")
