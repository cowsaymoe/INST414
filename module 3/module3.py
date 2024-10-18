import csv
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns


def main():
    # Import movies from dataset
    movies = pd.read_csv('data.csv')

    # only keep the columns we need
    movies = movies[['Series_Title', 'Released_Year', 'Genre', 'IMDB_Rating', 'Director', 'Gross']]

    # convert genres to a list
    movies['Genre'] = movies['Genre'].str.split(', ')
    # convert Gross to int
    movies['Gross'] = movies['Gross'].str.replace(',', '')

    # remove nan values
    movies = movies.dropna()

    # convert Gross to int
    movies['Gross'] = movies['Gross'].astype(int)

    print(movies.head())

    # Query items:
    query1 = "Avatar"
    query2 = "Avengers: Endgame"
    query3 = "Titanic"

    similar_movies = calculate_similarity(movies)

    # get the top 10 most similar movies to query 1
    top_similar_avatar = get_top_similar_movies(query1, similar_movies)
    print(f"Top 10 most similar movies to {query1}:")
    print(top_similar_avatar)


    # get the top 10 most similar movies to query 2
    top_similar_avengars = get_top_similar_movies(query2, similar_movies)
    print(f"Top 10 most similar movies to {query2}:")
    print(top_similar_avengars)

    # get the top 10 most similar movies to query 3
    top_similar_titanic = get_top_similar_movies(query3, similar_movies)
    print(f"Top 10 most similar movies to {query3}:")
    print(top_similar_titanic)

    # Graph
    plt.figure(figsize=(10, 10))
    sns.heatmap(similar_movies, annot=True, cmap='coolwarm')
    plt.title('Cosine Similarity Between Movies')
    plt.show()

def calculate_similarity(df):
    # Create feature vectors based on normalized columns
    features = df[['Gross', 'IMDB_Rating']].values
    similarity_matrix = cosine_similarity(features)
    
    # Create DataFrame for easier viewing
    similarity_df = pd.DataFrame(similarity_matrix, index=df['Series_Title'], columns=df['Series_Title'])
    return similarity_df


def get_top_similar_movies(query_movie, similarity_df, top_n=10):
    # Sort similar movies for the given query movie
    similar_movies = similarity_df[query_movie].sort_values(ascending=False)[1:top_n+1]
    return similar_movies

if __name__ == '__main__':
    main()