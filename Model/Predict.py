import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib
import os

# ---------------- CONFIG ----------------
SEQ_LEN = 30
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TARGET_COL = "Close"
MODEL_SAVE_PATH = "Model/trained_arima_dl_model.pt"
SCALER_PATH = "Model/scaler.save"
DATA_PATH = "Data/Processed/merged_stock_sentiment_data.csv"
OUTPUT_CSV = "Data/predicted_nextday.csv"

print("Using:", DEVICE)

# ---------------- LOAD DATA ----------------
df = pd.read_csv(DATA_PATH)
numeric_cols = [
    'Open', 'High', 'Low', 'Close', 'Volume',
    'SMA_14', 'EMA_14', 'RSI_14', 'MACD', 'MACD_signal', 'MACD_diff',
    'BB_Upper', 'BB_Lower', 'Momentum_10', 'OBV',
    'count_negative', 'count_positive', 'count_total',
    'positive_ratio', 'activity_count'
]

df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
df = df.dropna(subset=numeric_cols + [TARGET_COL]).reset_index(drop=True)
tickers = df['Ticker'].unique()

# ---------------- LOAD SCALER ----------------
scaler = joblib.load(SCALER_PATH)

# ---------------- DEFINE MODEL ----------------
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

# ---------------- LOAD MODEL ----------------
model = DeepARIMA(input_size=len(numeric_cols)).to(DEVICE)
model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))
model.eval()

# ---------------- PREDICTION ----------------
target_idx = numeric_cols.index(TARGET_COL)
results = []

for ticker in tickers:
    ticker_df = df[df['Ticker'] == ticker].copy()
    
    if len(ticker_df) < SEQ_LEN:
        print(f"Skipping {ticker}: insufficient data")
        continue

    last_seq = ticker_df[numeric_cols].tail(SEQ_LEN).values.astype(np.float32)
    last_seq_scaled = scaler.transform(last_seq)
    input_seq = torch.tensor(last_seq_scaled, dtype=torch.float32).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        pred_scaled = model(input_seq).item()

    # Inverse transform only the Close column
    dummy = np.zeros((1, len(numeric_cols)))
    dummy[0, target_idx] = pred_scaled
    pred_orig = scaler.inverse_transform(dummy)[0, target_idx]

    results.append({
        "Ticker": ticker,
        "Next_Day_Close": pred_orig
    })

# ---------------- SAVE RESULTS ----------------
results_df = pd.DataFrame(results)
results_df.to_csv(OUTPUT_CSV, index=False)
print(f"\n📈 Predictions saved to {OUTPUT_CSV}")
