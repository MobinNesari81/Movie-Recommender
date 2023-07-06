import streamlit as st
import pandas as pd
import numpy as np
from SVD_Model import Recommender_Model


@st.cache_resource
def load_model():
    return Recommender_Model()

model = load_model()


def view_suggestions(movies_array: list):
    movie1 = model.find_nearest_movie(movies_array[0])[0]
    movie2 = model.find_nearest_movie(movies_array[1])[0]
    movie3 = model.find_nearest_movie(movies_array[2])[0]
    suggestions = model.suggest([movie1, movie2, movie3])
    for movie in [movie1, movie2, movie3]:
        suggestions = suggestions[suggestions['title'] != movie]
    suggestions = suggestions.to_numpy()
    suggested_movies = []
    for row in suggestions:
        name = row[0]
        suggested_movies.append(model.get_movie_info(name))
    
    for idx, info in enumerate(suggested_movies):
        st.subheader(info['title'])
        st.markdown(f"__Overview:__ {info['overview']}")
        st.markdown(f"__Genres:__")
        st.write(info['genres'], value='genre')
        st.markdown(f"__Language:__ {info['language']}")
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
st.subheader("Please specify three movies which you like them")

with st.form("input_form"):
    movie1 = st.text_input('Movie 1', placeholder="Like: Ironman 1")
    movie2 = st.text_input('Movie 2', placeholder='Like: Ironman 2')
    movie3 = st.text_input('Movie 3', placeholder="Like: Ironman 3")
    
    
    submitted = st.form_submit_button("Submit")
    if submitted:
        view_suggestions([movie1, movie2, movie3])