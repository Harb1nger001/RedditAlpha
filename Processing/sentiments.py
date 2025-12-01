import os
import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# -------------------------------------------------------------
# Setup
# -------------------------------------------------------------

nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download('stopwords')

RAW_PATH = os.path.join("Data", "Raw", "reddit_posts.csv")
PROCESSED_DIR = os.path.join("Data", "Processed")
CLEANED_PATH = os.path.join(PROCESSED_DIR, "cleaned_reddit_posts.csv")
SENTIMENT_PATH = os.path.join(PROCESSED_DIR, "refined_reddit_sentiment_summary.csv")

os.makedirs(PROCESSED_DIR, exist_ok=True)

# -------------------------------------------------------------
# Text Cleaning
# -------------------------------------------------------------
def clean_text(text):
    """Preprocess text by removing URLs, special characters, and stopwords."""
    if not isinstance(text, str):
        return ""
    
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)

    tokens = word_tokenize(text)
    stop_words = set(stopwords.words("english"))
    filtered = [word for word in tokens if word not in stop_words]
    
    return " ".join(filtered)

# -------------------------------------------------------------
# Main Preprocessing + Sentiment Pipeline
# -------------------------------------------------------------
def process_reddit_data(input_path=RAW_PATH):
    print("📥 Loading raw Reddit data...")
    df = pd.read_csv(input_path)

    # Clean text columns
    print("\n🧹 Cleaning text fields...")
    tqdm.pandas()
    df['cleaned_title'] = df['title'].progress_apply(clean_text)
    df['cleaned_body'] = df['body'].progress_apply(clean_text)
    df['cleaned_comments'] = df['comments'].progress_apply(clean_text)

    # Save cleaned version
    df.to_csv(CLEANED_PATH, index=False)
    print(f"✔ Cleaned dataset saved to: {CLEANED_PATH}")

    # ---------------------------------------------------------
    # SENTIMENT ANALYSIS USING FINBERT
    # ---------------------------------------------------------
    print("\n⚙️ Loading FinBERT model...")
    model_name = "yiyanghkust/finbert-tone"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    def get_sentiment_label(text):
        if isinstance(text, str) and text.strip():
            inputs = tokenizer(text, truncation=True, max_length=512, return_tensors="pt")
            with torch.no_grad():
                logits = model(**inputs).logits

            probs = torch.softmax(logits, dim=1)[0]
            labels = model.config.id2label

            # Pick top label
            scores = [(labels[i], float(probs[i])) for i in range(len(probs))]
            top_label, top_score = max(scores, key=lambda x: x[1])

            # Override "Neutral"
            if top_label == "Neutral":
                return "Positive" if top_score >= 0.65 else "Negative"

            return top_label
        
        return "Negative"

    print("\n🔍 Running sentiment analysis on comments...")
    tqdm.pandas()
    df['sentiment'] = df['cleaned_comments'].progress_apply(get_sentiment_label)

    # ---------------------------------------------------------
    # AGGREGATION — DAILY SENTIMENT SUMMARY
    # ---------------------------------------------------------
    print("\n📊 Aggregating sentiment scores...")

    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['Date'] = df['timestamp'].dt.date

    sentiment_summary = (
        df.groupby(['Date', 'subreddit'])['sentiment']
        .value_counts()
        .unstack(fill_value=0)
        .reset_index()
    )

    sentiment_summary.rename(columns={
        'Positive': 'count_positive',
        'Negative': 'count_negative'
    }, inplace=True)

    sentiment_summary['count_total'] = sentiment_summary['count_positive'] + sentiment_summary['count_negative']
    sentiment_summary['positive_ratio'] = sentiment_summary['count_positive'] / sentiment_summary['count_total'].replace(0, 1)
    sentiment_summary['activity_count'] = sentiment_summary['count_total']

    # Save final output
    sentiment_summary.to_csv(SENTIMENT_PATH, index=False)

    print(f"\n✅ Sentiment summary saved to: {SENTIMENT_PATH}")
    print("🎉 Reddit preprocessing + sentiment pipeline complete!")

# -------------------------------------------------------------
# Run pipeline
# -------------------------------------------------------------
if __name__ == "__main__":
    process_reddit_data()
