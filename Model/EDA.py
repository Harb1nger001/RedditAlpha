import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

# ============================================================
# 📁 Setup
# ============================================================
input_path = "Data/Processed/merged_stock_sentiment_data.csv"
output_dir = "EDA_Report"
os.makedirs(output_dir, exist_ok=True)

print("📥 Loading dataset...")
df = pd.read_csv(input_path, parse_dates=["Date"])
df.sort_values("Date", inplace=True)

print(f"Loaded {len(df)} rows and {len(df.columns)} columns.")

# ============================================================
# 📝 1. Save Dataset Info
# ============================================================
print("📄 Saving dataset info...")

info_path = os.path.join(output_dir, "dataset_info.txt")

with open(info_path, "w") as f:
    f.write("DATASET INFO\n")
    f.write("="*60 + "\n\n")
    df.info(buf=f)
    f.write("\n\nDESCRIPTIVE STATISTICS\n")
    f.write("="*60 + "\n")
    f.write(df.describe().to_string())
    
print("✔ dataset_info.txt saved.")

# ============================================================
# ❓ 2. Missing Value Report
# ============================================================
print("🔍 Checking missing values...")

missing_df = df.isna().sum().reset_index()
missing_df.columns = ["Column", "MissingCount"]
missing_df["MissingPercent"] = (missing_df["MissingCount"] / len(df)) * 100

missing_path = os.path.join(output_dir, "missing_values.csv")
missing_df.to_csv(missing_path, index=False)

print("✔ Missing values report saved.")

# ============================================================
# 📊 3. Correlation Heatmap
# ============================================================
print("📊 Creating correlation heatmap...")

plt.figure(figsize=(14, 10))
num_df = df.select_dtypes(include=[np.number])
sns.heatmap(num_df.corr(), annot=False, cmap="coolwarm", linewidths=0.2)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig(f"{output_dir}/correlation_heatmap.png", dpi=300)
plt.close()

print("✔ Correlation heatmap saved.")

# ============================================================
# 📈 4. Time-Series Plots
# ============================================================
print("📈 Generating time-series plots...")

ts_cols = ["Close", "count_positive", "count_negative",
           "count_total", "positive_ratio", "activity_count"]

for col in ts_cols:
    if col in df.columns:
        plt.figure(figsize=(14, 6))
        sns.lineplot(data=df, x="Date", y=col)
        plt.title(f"{col} Over Time")
        plt.tight_layout()
        plt.savefig(f"{output_dir}/timeseries_{col}.png", dpi=300)
        plt.close()

print("✔ Time-series plots saved.")

# ============================================================
# 🔄 5. Rolling Correlation – Sentiment vs Stock Price
# ============================================================
print("🔄 Computing rolling correlations...")

if "Close" in df.columns and "positive_ratio" in df.columns:
    df["roll_corr_pos"] = df["Close"].rolling(30).corr(df["positive_ratio"])

    plt.figure(figsize=(14, 6))
    sns.lineplot(data=df, x="Date", y="roll_corr_pos")
    plt.title("30-Day Rolling Correlation: Positive Ratio vs Close Price")
    plt.axhline(0, color='black', linestyle="--")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/rolling_corr_pos.png", dpi=300)
    plt.close()

if "Close" in df.columns and "activity_count" in df.columns:
    df["roll_corr_activity"] = df["Close"].rolling(30).corr(df["activity_count"])

    plt.figure(figsize=(14, 6))
    sns.lineplot(data=df, x="Date", y="roll_corr_activity")
    plt.title("30-Day Rolling Correlation: Activity Count vs Close Price")
    plt.axhline(0, color='black', linestyle="--")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/rolling_corr_activity.png", dpi=300)
    plt.close()

print("✔ Rolling correlation plots saved.")

# ============================================================
# 🔥 8. Scatter + Trend Between Sentiment & Price
# ============================================================
print("🔥 Generating interaction plots...")

pairs = [
    ("positive_ratio", "Close"),
    ("activity_count", "Close"),
    ("count_total", "Close"),
]

for x, y in pairs:
    if x in df.columns and y in df.columns:
        plt.figure(figsize=(10, 6))
        sns.regplot(data=df, x=x, y=y, scatter_kws={"alpha": 0.3})
        plt.title(f"{x} vs {y}")
        plt.tight_layout()
        plt.savefig(f"{output_dir}/scatter_{x}_{y}.png", dpi=300)
        plt.close()

print("✔ Sentiment–stock interaction plots saved.")

print("\n🎉 EDA COMPLETE!")
print("All results saved to:", output_dir)
