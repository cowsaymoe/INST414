import os
from time import sleep
import praw
import pandas as pd
from datetime import datetime, timedelta
from textblob import TextBlob
import numpy as np



# Initialize Reddit API client
reddit = praw.Reddit(
    client_id='-GcMpHMS79wn71BDs89QEA',
    client_secret='qOXOgvB_kPmzPK8b5ZbnWw0_Tba2Og',
    user_agent='reddit-data-analysis'
)

def collect_posts():
    subreddit = reddit.subreddit('funny')
    posts_data = []
    
    for post in subreddit.top(time_filter='year', limit=1000):
        sleep(1)


        post_data = {
            'title': post.title,
            'created_utc': datetime.fromtimestamp(post.created_utc),
            'upvote_ratio': post.upvote_ratio,
            'num_comments': post.num_comments,
            'author_karma': post.author.link_karma if hasattr(post.author, 'link_karma') else 0,
            'author_age_days': (datetime.utcnow() - 
                datetime.fromtimestamp(post.created_utc)).days if post.author else 0,
            'final_score': post.score
        }
        posts_data.append(post_data)
        print(f"Posts collected: {len(posts_data)}")
    
    return pd.DataFrame(posts_data)


def engineer_features(df):
    # Title features
    df['title_length'] = df['title'].str.len()
    df['has_question'] = df['title'].str.contains('\?').astype(int)
    df['has_number'] = df['title'].str.contains('\d').astype(int)
    df['sentiment'] = df['title'].apply(lambda x: TextBlob(x).sentiment.polarity)
    
    # Timing features
    df['hour'] = df['created_utc'].dt.hour
    df['hour_sin'] = np.sin(2 * np.pi * df['hour']/24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour']/24)
    
    # Author features
    df['log_karma'] = np.log1p(df['author_karma'])
    
    return df

def main():
    print("Collecting posts.")
    posts_df = collect_posts()
    print(f"Collected {len(posts_df)} posts.")

    # saving to csv
    posts_df.to_csv('raw_posts.csv', index=False)
    
    print("Engineering features.")
    posts_df = engineer_features(posts_df)
    print("Features engineered.")
    
    print(posts_df.head())

    posts_df.to_csv('engineered_posts.csv', index=False)

if __name__ == "__main__":
    main()
