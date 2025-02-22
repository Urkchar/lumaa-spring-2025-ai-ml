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

# Set up vectorizer
vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = vectorizer.fit_transform(df["Keywords"])
user_vector = vectorizer.transform([user_input])

# Find similar movies
n = 5
similarities = cosine_similarity(user_vector, tfidf_matrix)
top_n_indices = np.argsort(similarities[0])[-n:][::-1]
top_n_similarities = similarities[0][top_n_indices]
top_n_titles = df.iloc[top_n_indices]["Title"].values
top_n_years = df.iloc[top_n_indices]["Year"].values

# Output
for title, score, year in zip(top_n_titles, top_n_similarities, top_n_years):
    print(f"{round(score * 100)}/100 - {title} ({year})")
