import streamlit as st
import pandas as pd
import numpy as np
from SVD_Model import Recommender_Model
from movie_poster import get_movie_poster_url


@st.cache_resource
def load_model():
    return Recommender_Model()

model = load_model()


def view_suggestions(movies_array: list):
    movie_titles = [movie.strip() for movie in movies_array if movie.strip()]
    if len(movie_titles) < 3:
        st.warning("Please enter at least three movie titles.")
        return
    suggestions = []
    for movie_title in movie_titles:
        nearest_movie = model.find_nearest_movie(movie_title)[0]
        suggestions.append(nearest_movie)
    suggested_movies = model.suggest(suggestions)
    for movie_title in movie_titles:
        suggested_movies = suggested_movies[suggested_movies['title'] != movie_title]
    suggested_movies = suggested_movies.to_numpy()
    for row in suggested_movies:
        name = row[0]
        info = model.get_movie_info(name)
        st.subheader(info['title'])
        st.markdown(f"__Overview:__ {info['overview']}")
        st.markdown(f"__Genres:__")
        st.write(info['genres'], value='genre')
        st.markdown(f"__Language:__ {info['language']}")
        poster_url = get_movie_poster_url(info['title'])
        st.image(poster_url, caption='Movie Poster')
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
    for i in range(num_movies):
        movie_inputs.append(st.text_input(f"Movie {i + 1}", key=f"movie_{i}"))
    submitted = st.form_submit_button("Submit")
    if submitted:
        view_suggestions(movie_inputs)
