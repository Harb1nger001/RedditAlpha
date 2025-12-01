import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ============================================================
# 📁 Ensure output directory exists
# ============================================================
output_dir = "Plots"
os.makedirs(output_dir, exist_ok=True)

# ============================================================
# 📥 Load pre-merged data
# ============================================================
merged_path = "Data/Processed/merged_stock_sentiment_data.csv"
merged_df = pd.read_csv(merged_path, parse_dates=["Date"])

print(f"Loaded merged dataset: {len(merged_df)} rows")
print("Saving all plots to:", output_dir)

merged_df.sort_values("Date", inplace=True)

# ============================================================
# 🎨 Seaborn Theme
# ============================================================
sns.set_theme(style="whitegrid")

# ============================================================
# 📊 1. Regular Sentiment Plots (Save instead of show)
# ============================================================
plt.figure(figsize=(18, 10))

# Subplot 1 — Positive & Negative Counts
plt.subplot(2, 2, 1)
sns.lineplot(data=merged_df, x="Date", y="count_positive", label="Positive")
sns.lineplot(data=merged_df, x="Date", y="count_negative", label="Negative")
plt.title("Positive vs Negative Sentiment Over Time")
plt.xlabel("Date")
plt.ylabel("Post Count")

# Subplot 2 — Total Posts
plt.subplot(2, 2, 2)
sns.lineplot(data=merged_df, x="Date", y="count_total")
plt.title("Total Posts Mentioning Stocks Over Time")
plt.xlabel("Date")
plt.ylabel("Total Post Count")

# Subplot 3 — Positive Ratio
plt.subplot(2, 2, 3)
sns.lineplot(data=merged_df, x="Date", y="positive_ratio")
plt.title("Positive Sentiment Ratio Over Time")
plt.xlabel("Date")
plt.ylabel("Positive Ratio")

# Subplot 4 — Reddit Activity
plt.subplot(2, 2, 4)
sns.lineplot(data=merged_df, x="Date", y="activity_count")
plt.title("Reddit Activity Over Time")
plt.xlabel("Date")
plt.ylabel("Activity Count")

plt.tight_layout()
plt.savefig(f"{output_dir}/sentiment_summary.png", dpi=300)
plt.close()

# ============================================================
# 📈 2. OVERLAY: Sentiment vs Stock Price (Dual Axis)
# ============================================================

# 🔥 Plot A: Positive Ratio vs Stock Price
plt.figure(figsize=(14, 8))
ax1 = sns.lineplot(data=merged_df, x="Date", y="positive_ratio", label="Positive Ratio")
ax2 = plt.twinx()
sns.lineplot(data=merged_df, x="Date", y="Close", ax=ax2, label="Stock Close", color="orange")

ax1.set_title("Positive Ratio vs Stock Price (Overlay)")
ax1.set_ylabel("Positive Ratio")
ax2.set_ylabel("Stock Price")

h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax1.legend(h1 + h2, l1 + l2, loc="upper left")

plt.tight_layout()
plt.savefig(f"{output_dir}/overlay_positive_ratio_stock.png", dpi=300)
plt.close()

# 🔥 Plot B: Activity Count vs Stock Price
plt.figure(figsize=(14, 8))
ax1 = sns.lineplot(data=merged_df, x="Date", y="activity_count", label="Activity Count")
ax2 = plt.twinx()
sns.lineplot(data=merged_df, x="Date", y="Close", ax=ax2, label="Stock Close", color="orange")

ax1.set_title("Reddit Activity vs Stock Price (Overlay)")
ax1.set_ylabel("Reddit Activity")
ax2.set_ylabel("Stock Price")

h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax1.legend(h1 + h2, l1 + l2, loc="upper left")

plt.tight_layout()
plt.savefig(f"{output_dir}/overlay_activity_stock.png", dpi=300)
plt.close()

print("\n✨ All plots saved successfully to:", output_dir)
