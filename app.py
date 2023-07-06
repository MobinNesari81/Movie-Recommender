import streamlit as st
import pandas as pd
from PIL import Image
from SVD_Model import Recommender_Model
from movie_poster import get_movie_poster_url


@st.cache_resource
def load_model():
    return Recommender_Model()

model = load_model()


def view_suggestions(movies_array: list, ratings: list):
    movie_titles = [movie.strip() for movie in movies_array if movie.strip()]
    if len(movie_titles) < 3:
        st.warning("Please enter at least three movie titles.")
        return
    if len(ratings) != len(movie_titles):
        st.warning("Please enter a rating for each movie.")
        return
    ratings = [float(rating) for rating in ratings]
    if not all(0 <= rating <= 5 for rating in ratings):
        st.warning("Please enter a rating between 0 and 5 for each movie.")
        return
    similar_users = model.get_similar_users(movie_titles, ratings)
    suggestions = model.suggest(similar_users)
    for movie_title in movie_titles:
        suggestions = suggestions[suggestions['title'] != movie_title]
    suggested_movies = suggestions.to_numpy()
    for row in suggested_movies:
        name = row[0]
        info = model.get_movie_info(name)
        st.subheader(info['title'])
        st.markdown(f"__Overview:__ {info['overview']}")
        st.markdown(f"__Genres:__")
        st.write(info['genres'], value='genre')
        st.markdown(f"__Language:__ {info['language']}")
        try:
            poster_url = get_movie_poster_url(info['title'])
            st.image(poster_url, caption='Movie Poster')
        except:
            st.error('Failed to retrieve movie poster.')
        st.divider()


with st.sidebar:
    logo = Image.open("MM Logo.jpeg")
    st.image(logo, caption='MM Movie Recommender')
    st.title("MM Movie Recommender")
    st.subheader("Development Team:")
    st.markdown("<a href='https://www.linkedin.com/in/mobin-nesari/'>Mobin Nesari</a>", unsafe_allow_html=True)
    st.markdown("<a href='https://www.linkedin.com/in/m0hsnn/'>Seyed Mohsen Sadeghi</a>", unsafe_allow_html=True)


st.title("MM Movie Recommender")
st.header("Movie Names:")
st.subheader("Please enter at least three movies that you like.")

with st.form("input_form"):
    num_movies = st.number_input(
        "Number of movies",
        min_value=3,
        max_value=10,
        value=3,
        step=1,
    )
    movie_inputs = []
    rating_inputs = []
    for i in range(num_movies):
        movie_inputs.append(st.text_input(f"Movie {i + 1}", key=f"movie_{i}"))
        rating_inputs.append(st.number_input(f"Rating for Movie {i + 1}", min_value=0, max_value=5, value=5, step=0.5))
    submitted = st.form_submit_button("Submit")
    if submitted:
        view_suggestions(movie_inputs, rating_inputs)
