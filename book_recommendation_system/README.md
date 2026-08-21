# 📚 Book Recommendation System

A collaborative-filtering based book recommendation engine, built as a
complete Python project. Includes three recommendation algorithms, a
CLI, a web app, and a documented Jupyter notebook.

## Features

- **User-Based Collaborative Filtering** — recommends books liked by users with similar taste
- **Item-Based Collaborative Filtering** — recommends books similar to ones you rated highly
- **SVD Matrix Factorization** — latent-factor model (the Netflix Prize technique)
- **Hybrid** — blends SVD and item-based scores
- RMSE evaluation on a held-out test split
- Interactive **Streamlit** web app
- CLI for quick terminal use
- Jupyter notebook with full explanations, for reports/presentations

## Project Structure

```
book_recommendation_system/
├── data/
│   ├── books.csv          # book_id, title, author, genre
│   └── ratings.csv        # user_id, book_id, rating
├── src/
│   ├── data_loader.py     # loads/generates data, builds the rating matrix
│   ├── recommender.py     # the 4 recommendation algorithms
│   └── evaluate.py        # RMSE evaluation
├── notebooks/
│   └── book_recommendation_system.ipynb
├── main.py                 # CLI entry point
├── app.py                   # Streamlit web app
├── requirements.txt
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
```

The `data/` folder already ships with a synthetic sample dataset
(40 users, 25 books, 400 ratings) so everything runs immediately.

### Using your own data

Replace `data/books.csv` and `data/ratings.csv` with a real dataset
(e.g. **Book-Crossing** or **Goodbooks-10k**), keeping these columns:

- `books.csv`: `book_id, title, author, genre`
- `ratings.csv`: `user_id, book_id, rating` (1-5 scale)

## Usage

### CLI

```bash
python main.py                              # recommend for user 1, all 4 methods
python main.py --user 7                     # recommend for a specific user
python main.py --user 7 --method svd --top_n 10
python main.py --evaluate                   # print SVD model RMSE
```

### Web App

```bash
streamlit run app.py
```

Opens an interactive UI where you can pick a user, a recommendation
method, and see results in a table.

### Notebook

Open `notebooks/book_recommendation_system.ipynb` in Jupyter — it
walks through the data, each algorithm, and the evaluation step by
step, with explanations. Good for a project report or presentation.

## How It Works

1. **Data → Matrix**: ratings are pivoted into a user × book matrix
   (rows = users, columns = books, cells = ratings, blank = unrated).
2. **User-Based CF**: computes cosine similarity between users based on
   their rating patterns, then predicts a book's rating as a
   similarity-weighted average of what similar users rated it.
3. **Item-Based CF**: computes cosine similarity between books based on
   how all users rated them, then predicts a rating as a
   similarity-weighted average of the user's own ratings on similar books.
4. **SVD**: factorizes the matrix into low-rank user-factor and
   item-factor matrices representing latent "taste dimensions";
   multiplying them back together fills in predicted ratings.
5. **Evaluation**: 20% of ratings are held out as a test set; RMSE
   measures how far off the SVD model's predictions are.

## Extending the Project

- Swap in a real dataset (see above)
- Add content-based filtering using book descriptions/genres (TF-IDF)
- Add a "cold start" fallback (recommend most-popular books for new users)
- Deploy the Streamlit app (Streamlit Community Cloud, Render, etc.)
