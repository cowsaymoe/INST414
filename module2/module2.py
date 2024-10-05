"""
Process data and generates graphs
"""

from matplotlib import pyplot as plt
import praw, os
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
import networkx as nx

load_dotenv()

CLIENT_ID = os.getenv("REDDIT_ID")
CLIENT_SECRET = os.getenv("REDDIT_SECRET")

def collect_data(subreddit_name: str, time_period: str = "all", limit: int = 10) -> pd.DataFrame:
    """
    Collects data from a subreddit for a given time period.

    Args:
        subreddit_name (str): The name of the subreddit.
        time_period (str): The time period to collect data for.
    Returns:
        pd.DataFrame: A DataFrame containing the collected data.
    """
    # Initialize the Reddit API client
    reddit = praw.Reddit(
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET,
        user_agent="Reddit Data Collection"
    )

    # Specify the subreddit and time period
    subreddit_name = subreddit_name
    time_period = time_period

    # Create empty lists to store the data
    data = []

    # Collect data from the subreddit
    subreddit = reddit.subreddit(subreddit_name)
    for submission in subreddit.top(time_filter=time_period, limit=limit):
        print(f"Collecting data for post: {submission.title}")
        submission.comments.replace_more(limit=0)
        for comment in submission.comments.list():
            if comment.author:
                data.append({
                    "post_id": submission.id,
                    "post_author": submission.author.name if submission.author else "[deleted]",
                    "post_score": submission.score,
                    "post_title": submission.title,
                    "comment_id": comment.id,
                    "comment_author": comment.author.name if comment.author else "[deleted]",
                    "comment_score": comment.score,
                    "timestamp": datetime.fromtimestamp(comment.created_utc)
                })

    # Convert the data to a DataFrame
    df = pd.DataFrame(data)
    return df

def graph_data(data: pd.DataFrame):
    """
    Graphs the data to identify the top 3 important nodes.

    Args:
        data (pd.DataFrame): The DataFrame containing the data.
    """

    # Create a graph from our data
    G = nx.Graph()

    # Add edges to the graph
    for _, row in data.iterrows():
        G.add_edge(row['post_author'], row['comment_author'])

    # Calculate centrality measures
    degree_centrality = nx.degree_centrality(G)
    eigenvector_centrality = nx.eigenvector_centrality_numpy(G)

    # Calculate engagement score
    engagement_scores = data.groupby('comment_author').agg({
        'comment_id': 'count',
        'comment_score': 'mean'
    }).reset_index()
    engagement_scores['engagement_score'] = engagement_scores['comment_id'] * engagement_scores['comment_score']

    # Combine metrics
    importance_scores = pd.DataFrame({
        'user': list(G.nodes()),
        'degree_centrality': [degree_centrality[node] for node in G.nodes()],
        'eigenvector_centrality': [eigenvector_centrality[node] for node in G.nodes()]
    })
    importance_scores = importance_scores.merge(engagement_scores, left_on='user', right_on='comment_author', how='left')
    importance_scores['importance'] = (
        importance_scores['degree_centrality'] +
        importance_scores['eigenvector_centrality'] +
        importance_scores['engagement_score'].fillna(0)
    )

    # Identify top 3 important nodes
    top_3_important_nodes = importance_scores.nlargest(3, 'importance')
    
    print("Top 3 important nodes:")
    print(top_3_important_nodes)

    # Plot the graph
    pos = nx.spring_layout(G, weight='degree_centrality', k=0.5)

    node_color_map = []
    for node in G.nodes():
        if node in top_3_important_nodes['user'].values:
            node_color_map.append('red')
        else:
            node_color_map.append('skyblue')
    
    plt.figure(figsize=(12, 8))

    nx.draw_networkx_edges(G, pos, alpha=0.3)


    nx.draw_networkx_nodes(G, pos, node_color=node_color_map, node_size=500, alpha=0.8)

    nx.draw_networkx_labels(G, pos, font_size=8, font_color='black')

    plt.title("Reddit User Interaction Network")
    plt.tight_layout()
    plt.show()

    

def main():
    print("Collecting data from Reddit...")
    data = collect_data("AskReddit", "day", 1)
    print(data.head())

    print("Graphing data...")
    graph_data(data)



if __name__ == '__main__':
    main()