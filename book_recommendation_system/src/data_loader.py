"""
Data loading utilities for the Book Recommendation System.

By default, loads the sample books.csv / ratings.csv shipped in the
data/ folder. To use your own dataset (e.g. Book-Crossing,
Goodbooks-10k), just replace those two CSV files -- as long as they
have these columns, everything else works unchanged:

    books.csv   -> book_id, title, author, genre
    ratings.csv -> user_id, book_id, rating   (rating scale 1-5)
"""

import os
import numpy as np
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
BOOKS_PATH = os.path.join(DATA_DIR, "books.csv")
RATINGS_PATH = os.path.join(DATA_DIR, "ratings.csv")


def generate_sample_data(n_users=40, n_books=25, n_ratings=400, seed=42):
    """Generates a small synthetic (books, ratings) dataset. Used to
    create the sample CSVs shipped with this project, and as a
    fallback if the CSV files are ever missing."""
    rng = np.random.default_rng(seed)

    genres = ["Fantasy", "Mystery", "Romance", "Sci-Fi", "Non-Fiction"]
    titles = [
        "The Silent Forest", "Shadows of Time", "Love in Autumn",
        "The Last Algorithm", "Whispering Stars", "Blood Moon Rising",
        "The Paper Kingdom", "Beyond the Horizon", "Echoes of War",
        "The Quiet Storm", "City of Glass", "Winter's Promise",
        "The Forgotten Path", "Rise of the Machines", "Golden Hour",
        "The Ocean's Secret", "Midnight Library", "The Iron Crown",
        "Letters to Nowhere", "The Glass House", "Fire and Ash",
        "The Wandering Mind", "Song of the Sea", "Broken Compass",
        "The Last Chapter",
    ]

    books = pd.DataFrame({
        "book_id": range(1, n_books + 1),
        "title": titles[:n_books],
        "author": [f"Author {chr(65 + i % 20)}" for i in range(n_books)],
        "genre": [genres[i % len(genres)] for i in range(n_books)],
    })

    user_pref = rng.choice(genres, size=n_users)
    rows = []
    seen = set()
    while len(rows) < n_ratings:
        u = rng.integers(1, n_users + 1)
        b = rng.integers(1, n_books + 1)
        if (u, b) in seen:
            continue
        seen.add((u, b))
        base = 3.0
        if books.loc[b - 1, "genre"] == user_pref[u - 1]:
            base += 1.5
        rating = np.clip(rng.normal(base, 0.9), 1, 5)
        rows.append((u, b, round(rating)))

    ratings = pd.DataFrame(rows, columns=["user_id", "book_id", "rating"])
    return books, ratings


def load_data():
    """Loads books/ratings from the data/ CSVs, generating them first
    if they don't exist yet."""
    if not (os.path.exists(BOOKS_PATH) and os.path.exists(RATINGS_PATH)):
        books, ratings = generate_sample_data()
        os.makedirs(DATA_DIR, exist_ok=True)
        books.to_csv(BOOKS_PATH, index=False)
        ratings.to_csv(RATINGS_PATH, index=False)
    else:
        books = pd.read_csv(BOOKS_PATH)
        ratings = pd.read_csv(RATINGS_PATH)
    return books, ratings


def build_matrix(ratings, books):
    """Pivots ratings into a user x book matrix (NaN = not rated)."""
    n_users = ratings.user_id.max()
    matrix = ratings.pivot_table(
        index="user_id", columns="book_id", values="rating"
    ).reindex(index=range(1, int(n_users) + 1), columns=sorted(books.book_id.unique()))
    return matrix
