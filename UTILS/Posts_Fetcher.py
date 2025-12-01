import os
import praw
import pandas as pd
import datetime as dt
import time

# Initialize Reddit API (Replace with your credentials)
reddit = praw.Reddit(
    client_id='K5eClhs10PzsecZNXB-h2Q',
    client_secret='4iKyc6eLa8xf2gN4es6sl0hv9gL-Bw',
    user_agent='your_user_agent'
)

def fetch_relevant_posts(subreddits, keywords, start_date, limit=1000):
    """Fetch relevant posts using PRAW's search, with a wait between subreddits."""
    posts = []
    start_timestamp = int(dt.datetime.strptime(start_date, "%Y-%m-%d").timestamp())

    for subreddit in subreddits:
        print(f"Fetching posts from r/{subreddit}...")

        try:
            for post in reddit.subreddit(subreddit).search(
                    " OR ".join(keywords),
                    sort="new",
                    time_filter="all",
                    limit=limit):

                if post.created_utc < start_timestamp:
                    continue

                post.comments.replace_more(limit=0)
                comments = [comment.body for comment in post.comments[:100]]
                comments_text = " ||| ".join(comments)

                posts.append([
                    subreddit,
                    post.title,
                    post.selftext,
                    post.score,
                    post.num_comments,
                    dt.datetime.fromtimestamp(post.created_utc),
                    post.url,
                    comments_text
                ])

                time.sleep(1)

        except Exception as e:
            print(f"Error fetching data from r/{subreddit}: {e}")

        print(f"Finished fetching r/{subreddit}. Waiting before next subreddit...")
        time.sleep(100)

    return pd.DataFrame(
        posts,
        columns=[
            'subreddit', 'title', 'body', 'score', 'num_comments',
            'timestamp', 'url', 'comments'
        ]
    )

def save_to_csv(df, filename='reddit_posts.csv'):
    """Save the DataFrame to Data/Raw/."""
    os.makedirs("Data/Raw", exist_ok=True)
    filepath = os.path.join("Data/Raw", filename)
    df.to_csv(filepath, index=False)
    print(f"Saved {len(df)} posts to {filepath}")

# Example usage
if __name__ == "__main__":
    subreddits = [
        'IndianStockMarket', 'IndiaInvestments', 'StockMarketIndia',
        'IndianStreetBets', 'stocks', 'investing', 'financialindependence'
    ]

    keywords = [
        'Reliance', 'Tata', 'Infosys', 'HDFC', 'Nifty', 'ICICI',
        'Wipro', 'Adani', 'Maruti', 'Larsen'
    ]

    start_date = (dt.datetime.now() - dt.timedelta(days=16*365)).strftime('%Y-%m-%d')

    reddit_df = fetch_relevant_posts(subreddits, keywords, start_date, limit=3000)
    save_to_csv(reddit_df, filename="reddit_posts.csv")
