import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Get data
user_input = ""
while not user_input:
    user_input = input("Give a short text description of your movie preferences to be recommended 5 movies.\n")
df = pd.read_csv("data.csv")

# Clean data
user_input = re.sub(r"[^\w\s]", "", user_input).lower()
df["Keywords"] = df["Keywords"].apply(lambda x: re.sub(r"[^\w\s]", "", x).lower())
# print(df.shape)
# print(df.head())

# Set up vectorizer
vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = vectorizer.fit_transform(df["Keywords"])
user_vector = vectorizer.transform([user_input])
# print(f"tfidf_matrix.shape: {tfidf_matrix.shape}")

# Convert data to vectors
# tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out())
# tfidf_df = pd.DataFrame(tfidf_matrix.toarray())
# print(f"tfidf_df.shape: {tfidf_df.shape}")
# vector_df = pd.DataFrame(tfidf_matrix.toarray())
# print(f"vector_df.shape: {vector_df.shape}")

# Convert user input to vectors

# print(user_vector)
# Find similar movies
n = 5
similarities = cosine_similarity(user_vector, tfidf_matrix)
top_n_indices = np.argsort(similarities[0])[-n:][::-1]
# print(f"similarities.shape: {similarities.shape}")
# print(similarities)

# print(f"user_vector.shape: {user_vector.shape}")
# print(f"vector_df.shape: {vector_df.shape}")
# print(f"top_5_indices.shape: {top_5_indices.shape}")
# top_5_vectors = tfidf_matrix[top_5_indices]
# print(f"top_5_vectors.shape: {top_5_vectors.shape}")
# print(type(top_5_vectors))
# print(top_5_vectors)
# # print(top_5_vectors)
# print(top_5_vectors.toarray())
# top_n_movies = df.iloc[top_n_indices]
top_n_similarities = similarities[0][top_n_indices]
# print(top_5_movies)
top_n_titles = df.iloc[top_n_indices]["Title"].values
top_n_years = df.iloc[top_n_indices]["Year"].values
# print(top_5_titles)
for title, score, year in zip(top_n_titles, top_n_similarities, top_n_years):
    # print(type(year))
    print(f"{round(score * 100)}/100 - {title} ({year})")
