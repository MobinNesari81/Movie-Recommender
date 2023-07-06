import requests

def get_movie_poster_url(title_or_id, api_key='a2e7ab47', id=False):
    if id:
        url = f"http://www.omdbapi.com/?apikey={api_key}&i={title_or_id}"
    else:
        url = f"http://www.omdbapi.com/?apikey={api_key}&t={title_or_id}"

    response = requests.get(url)

    if response.status_code == 200:
        movie_data = response.json()
        poster_url = movie_data.get("Poster", "nan")
        return poster_url
    else:
        print(f"Error retrieving movie data for '{title_or_id}': {response.text}")
        return None
