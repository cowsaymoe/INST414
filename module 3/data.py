
import json
from bs4 import BeautifulSoup

import requests

def main():

    # Import movies from dataset
    
    dataset = []

    with open("imdb_movies_2000to2022.prolific.json", "r") as f:
        for line in f:
            dataset.append(json.loads(line))



    # Query items:
    avatar = "tt0499549"
    avengers_endgame = "tt4154796"
    titanic = "tt0120338"

    box_office = get_boxoffice(avatar)

    with open("dataset.csv", "w") as f:
    # Find the box office for each movie
        for movie in dataset:
            box_office = get_boxoffice(movie["imdb_id"])
            movie["box_office"] = box_office
            f.write(json.dumps(movie) + "\n")

    print("Dataset with box office data:")
    print(dataset)

    # save dataset



    # find the top 10 most similar movies using cosine similarity

def get_boxoffice(movie):
    base_url = 'https://www.boxofficemojo.com/title/'
    url = base_url + movie

    response = requests.get(url)

    if response.status_code != 200:
        print(f"Failed to fetch {url}")
        return None

    soup = BeautifulSoup(response.content, 'html.parser')

    box_office = None

    try:
        box_office = soup.find('span', class_='money').text.replace('$', '').replace(',', '')
    except AttributeError:
        print(f"Could not find box office data for {movie}")
    return float(box_office) if box_office else None

if __name__ == "__main__":
    main()