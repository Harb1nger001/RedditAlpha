# RedditAlpha 🚀

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.13-red?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

RedditAlpha is a data-driven **next-day stock price prediction** tool for Indian equities, merging **Reddit sentiment analysis** with **historical stock data** and **technical indicators**. It leverages **FinBERT** for sentiment scoring and a custom **Deep ARIMA-inspired neural network** for forecasting.

---

## ⚡ Features

- Collects 16 years of Reddit posts & comments from top Indian finance subreddits.  
- Integrates historical stock data with technical indicators.  
- Performs sentiment analysis with FinBERT on Reddit discussions.  
- Trains a deep ARIMA-inspired model for **next-day stock prediction**.  
- Provides per-ticker evaluation metrics for model performance.

---

## 📊 Evaluation Metrics

**Overall Test Metrics:**

| Metric | Value |
|--------|-------|
| RMSE   | 0.00286 |
| MAE    | 0.00138 |
| R²     | 0.99973 |

**Per-Ticker Performance:**

| Ticker        | RMSE   | MAE    | R²     |
|---------------|--------|--------|--------|
| ADANIENT.NS   | 0.0021 | 0.0013 | 0.9979 |
| HDFCBANK.NS   | 0.0009 | 0.0005 | 0.9909 |
| ICICIBANK.NS  | 0.0009 | 0.0006 | 0.9958 |
| INFY.NS       | 0.0010 | 0.0007 | 0.9977 |
| LT.NS         | 0.0017 | 0.0011 | 0.9981 |
| MARUTI.NS     | 0.0054 | 0.0036 | 0.9980 |
| RELIANCE.NS   | 0.0014 | 0.0007 | 0.9937 |
| TATASTEEL.NS  | 0.0010 | 0.0007 | 0.7114 |
| WIPRO.NS      | 0.0007 | 0.0005 | 0.9324 |
| ^NSEI         | 0.0064 | 0.0042 | 0.9992 |

---

## 🔧 Tech Stack

- **Python 3.11**  
- **PyTorch** for deep learning  
- **FinBERT** for sentiment analysis  
- **Pandas & NumPy** for data processing  
- **Scikit-learn** for scaling & metrics  

---

### ⚡ Highlights

- Merges **Reddit sentiment** and **technical indicators** for more informed predictions.  
- Generates **per-ticker evaluation metrics**, enabling granular insight.  
- Modular codebase ready for extensions like multi-step forecasting.

---

*Made with ❤️ and ☕ by a passionate data enthusiast.*
