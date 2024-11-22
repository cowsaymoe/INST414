import requests
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def fetch_movie_data(api_key):
    base_url = "https://api.themoviedb.org/3"
    movies = []
    
    for year in range(2010, 2024):
        url = f"{base_url}/discover/movie"
        params = {
            'api_key': api_key,
            'primary_release_year': year,
            'sort_by': 'vote_count.desc',
            'limit': 100
        }
        
        response = requests.get(url, params=params)
        data = response.json()
        movies.extend(data['results'])
    
    return pd.DataFrame(movies)


def main():
    print("Getting movies.")
    # Fetch and clean data
    df = fetch_movie_data('092846a2bc9abc84690374f21b5b9c99')
    print(f"Got {len(df)} movies.")
    # priunt all columns
    print(df.columns)

    df['budget'].fillna(df.groupby('release_date')['popularity'].transform('median'), inplace=True)

    # One-hot encode genres
    genres = df['genres'].str.get_dummies(sep='|')
    df = pd.concat([df, genres], axis=1)

    # Clean runtime
    clean_df = df[(df['runtime'] >= 60) & (df['runtime'] <= 240)]

    # Prepare features
    features = ['runtime', 'vote_average', 'revenue_ratio']
    features.extend(genres.columns)

    # Normalize numerical features
    scaler = StandardScaler()
    clean_df[features] = scaler.fit_transform(clean_df[features])

    

    inertias = []
    k_range = range(1, 11)

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(clean_df[features])
        inertias.append(kmeans.inertia_)

    plt.plot(k_range, inertias, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Inertia')
    plt.title('Elbow Method For Optimal k')
    plt.show()



if __name__ == '__main__':
    main()