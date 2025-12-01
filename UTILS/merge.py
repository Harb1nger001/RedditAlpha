import pandas as pd
import os


def load_and_prepare(path, date_col="Date"):
    """Load CSV, parse date, normalize timezone, drop duplicates safely."""
    df = pd.read_csv(path, parse_dates=[date_col])

    # Normalize timezone + strip to date
    df[date_col] = (
        pd.to_datetime(df[date_col], utc=True)
        .dt.tz_localize(None)
        .dt.normalize()
    )

    # Drop duplicates correctly
    if "Ticker" in df.columns:
        df = df.drop_duplicates(subset=[date_col, "Ticker"])
    else:
        df = df.drop_duplicates(subset=[date_col])

    # Sort correctly
    if "Ticker" in df.columns:
        df = df.sort_values(["Ticker", date_col])
    else:
        df = df.sort_values(date_col)

    return df


# ---------------------- LOAD DATA ---------------------- #
print("📥 Loading stock data...")
stock_df = load_and_prepare("Data/Processed/stock_data_with_indicators.csv")

print("📥 Loading Reddit sentiment data...")
reddit_df = load_and_prepare("Data/Processed/refined_reddit_sentiment_summary.csv")

# Ensure Reddit is sorted for asof
reddit_df = reddit_df.sort_values("Date")

print("\n📌 Tickers detected in stock data:")
print(stock_df["Ticker"].unique())


# ------------------ MERGE PER TICKER ------------------- #
print("\n🔗 Merging stock + sentiment per ticker...")

merged_list = []

for ticker in stock_df["Ticker"].unique():
    print(f"   • Processing {ticker} ...")

    temp_stock = stock_df[stock_df["Ticker"] == ticker].sort_values("Date")

    # ASOF merge: backward = take latest sentiment up to that date
    merged_temp = pd.merge_asof(
        temp_stock,
        reddit_df,
        on="Date",
        direction="backward",
    )

    # Forward fill leading sentiment gaps
    merged_temp.ffill(inplace=True)

    merged_list.append(merged_temp)


# Combine all tickers together
merged_df = pd.concat(merged_list, ignore_index=True)

print(f"\n✅ Merge complete!")
print(f"   Stock rows     : {len(stock_df)}")
print(f"   Final rows     : {len(merged_df)}")
print("   (these should match)")


# ----------------------- SAVE -------------------------- #
output_path = "Data/Processed/merged_stock_sentiment_data.csv"
os.makedirs("Data/Processed", exist_ok=True)

merged_df.to_csv(output_path, index=False)

print(f"\n💾 Saved merged dataset → {output_path}")
