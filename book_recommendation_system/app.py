"""
Streamlit web app for the Book Recommendation System.

Run with:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd

from src.data_loader import load_data, build_matrix
from src.recommender import (
    user_based_recommend,
    item_based_recommend,
    svd_recommend,
    hybrid_recommend,
)

st.set_page_config(
    page_title="Book Recommendation System",
    page_icon="📚",
    layout="centered"
)

st.title("📚 Book Recommendation System")
st.caption(
    "Collaborative filtering demo — User-Based, Item-Based, SVD, and Hybrid"
)


@st.cache_data
def get_data():
    return load_data()


books, ratings = get_data()
matrix = build_matrix(ratings, books)


METHODS = {
    "User-Based CF": user_based_recommend,
    "Item-Based CF": item_based_recommend,
    "SVD (Matrix Factorization)": svd_recommend,
    "Hybrid (SVD + Item-Based)": hybrid_recommend,
}


with st.sidebar:
    st.header("Settings")

    user_id = st.selectbox(
        "Select a user",
        options=list(matrix.index),
        index=0
    )

    method_name = st.radio(
        "Recommendation method",
        list(METHODS.keys())
    )

    top_n = st.slider(
        "Number of recommendations",
        min_value=3,
        max_value=15,
        value=5
    )


st.subheader(f"Books user {user_id} already rated")

already_rated = matrix.loc[user_id].dropna()

rated_books = books[
    books.book_id.isin(already_rated.index)
].copy()

rated_books["rating"] = rated_books.book_id.map(already_rated)

st.dataframe(
    rated_books[["title", "author", "genre", "rating"]],
    hide_index=True
)


st.subheader(
    f"Recommended for user {user_id} — {method_name}"
)

fn = METHODS[method_name]

results = fn(
    matrix,
    user_id,
    books,
    top_n=top_n
)


if results:
    result_df = pd.DataFrame(results)[
        ["title", "author", "genre", "predicted_rating"]
    ]

    st.dataframe(
        result_df,
        hide_index=True
    )
else:
    st.info(
        "No recommendations available for this user with the current settings."
    )


st.divider()

st.subheader("😊 Choose Your Mood")


mood_books = {

    "😊 Happy": [
        "Wonder",
        "Matilda",
        "The Secret Garden",
        "Anne of Green Gables",
        "Eleanor Oliphant Is Completely Fine"
    ],

    "😢 Sad": [
        "The Kite Runner",
        "Five Feet Apart",
        "All the Bright Places",
        "The Book Thief",
        "A Man Called Ove"
    ],

    "💪 Motivational": [
        "Atomic Habits",
        "Can't Hurt Me",
        "The Power of Habit",
        "You Can Win",
        "Think and Grow Rich"
    ],

    "❤️ Romantic": [
        "The Notebook",
        "Me Before You",
        "It Ends with Us",
        "The Love Hypothesis",
        "Pride and Prejudice"
    ],

    "😌 Peaceful": [
        "Ikigai",
        "The Power of Now",
        "The Art of Happiness",
        "A New Earth"
    ],

    "🧠 Self-Improvement": [
        "Deep Work",
        "Think Like a Monk",
        "The Psychology of Money",
        "The 7 Habits of Highly Effective People"
    ],

    "💻 Tech": [
        "Clean Code",
        "The Pragmatic Programmer",
        "Python Crash Course",
        "Automate the Boring Stuff with Python",
        "Head First Java",
        "Hands-On Machine Learning"
    ]
}


mood = st.selectbox(
    "How are you feeling today?",
    list(mood_books.keys())
)


st.write("### 📚 Recommended Books")


for book in mood_books[mood]:
    st.write("📖", book)


st.divider()


st.caption(
    "Dataset: synthetic sample data (data/books.csv, data/ratings.csv). "
    "Replace those CSVs with a real dataset (e.g. Book-Crossing, Goodbooks-10k) "
    "using the same column format to use real data."
)