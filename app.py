#Import libraries
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

#Load movies
movies = pd.read_csv("ml-latest-small/movies.csv")

#Prepare features
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['genres'])

#Compute similarity
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

#Recommendation function
def recommend(movie_title, cosine_sim=cosine_sim):
    # Convert input and dataset titles to lowercase
    movie_title = movie_title.lower()
    movies_lower = movies['title'].str.lower()
    
    # Find movies
    matches = movies_lower[movies_lower.str.contains(movie_title)]
    if matches.empty:
        return ["Movie not found!"]
    
    # Take the first match
    idx = matches.index[0]
    
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:6]  # top 5
    movie_indices = [i[0] for i in sim_scores]
    return movies['title'].iloc[movie_indices].tolist()
# Streamlit UI
st.title("Movie Recommendation System")

movie_input = st.text_input("Enter a movie title:")

if st.button("Recommend"):
    recommendations = recommend(movie_input)
    st.write("Here are some movies you might like:")
    for rec in recommendations:
        st.write("----> " + rec)
