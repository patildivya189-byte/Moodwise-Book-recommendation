"""
Collaborative filtering recommendation algorithms:
  - User-based CF   (cosine similarity between users)
  - Item-based CF    (cosine similarity between books)
  - SVD Matrix Factorization (latent factor model)
"""

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD


def _attach_titles(ranked, books):
    out = []
    for book_id, score in ranked:
        row = books.loc[books.book_id == book_id].iloc[0]
        out.append({
            "book_id": int(book_id),
            "title": row["title"],
            "author": row["author"],
            "genre": row["genre"],
            "predicted_rating": round(float(score), 2),
        })
    return out


def user_based_recommend(matrix, user_id, books, top_n=5, k_neighbors=10):
    """Recommend books liked by the most similar users to this user."""
    filled = matrix.fillna(0)
    sim = cosine_similarity(filled)
    sim_df = pd.DataFrame(sim, index=matrix.index, columns=matrix.index)

    neighbors = sim_df[user_id].drop(user_id).sort_values(ascending=False)
    neighbors = neighbors[neighbors > 0].head(k_neighbors)

    user_rated = matrix.loc[user_id].dropna().index
    scores = {}
    for book_id in matrix.columns:
        if book_id in user_rated:
            continue
        num, den = 0.0, 0.0
        for neighbor_id, similarity in neighbors.items():
            r = matrix.loc[neighbor_id, book_id]
            if not np.isnan(r):
                num += similarity * r
                den += similarity
        if den > 0:
            scores[book_id] = num / den

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
    return _attach_titles(ranked, books)


def item_based_recommend(matrix, user_id, books, top_n=5):
    """Recommend books similar to the ones this user already rated highly."""
    filled = matrix.fillna(0)
    item_sim = cosine_similarity(filled.T)
    item_sim_df = pd.DataFrame(item_sim, index=matrix.columns, columns=matrix.columns)

    user_ratings = matrix.loc[user_id].dropna()
    scores = {}
    for book_id in matrix.columns:
        if book_id in user_ratings.index:
            continue
        sims = item_sim_df[book_id][user_ratings.index]
        num = (sims * user_ratings).sum()
        den = sims.abs().sum()
        if den > 0:
            scores[book_id] = num / den

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
    return _attach_titles(ranked, books)


def svd_recommend(matrix, user_id, books, top_n=5, n_components=8):
    """Recommend books using latent-factor matrix factorization (SVD)."""
    filled = matrix.fillna(matrix.stack().mean())
    n_components = min(n_components, min(filled.shape) - 1)

    svd = TruncatedSVD(n_components=n_components, random_state=42)
    user_factors = svd.fit_transform(filled)
    item_factors = svd.components_

    predicted = pd.DataFrame(
        user_factors @ item_factors, index=matrix.index, columns=matrix.columns
    )

    already_rated = matrix.loc[user_id].dropna().index
    preds = predicted.loc[user_id].drop(already_rated)
    ranked = preds.sort_values(ascending=False).head(top_n)
    return _attach_titles(list(ranked.items()), books)


def hybrid_recommend(matrix, user_id, books, top_n=5):
    """Simple hybrid: averages the SVD and item-based predicted scores
    for books recommended by either method, then re-ranks."""
    svd_scores = {r["book_id"]: r["predicted_rating"]
                  for r in svd_recommend(matrix, user_id, books, top_n=top_n * 3)}
    item_scores = {r["book_id"]: r["predicted_rating"]
                   for r in item_based_recommend(matrix, user_id, books, top_n=top_n * 3)}

    all_ids = set(svd_scores) | set(item_scores)
    combined = []
    for book_id in all_ids:
        parts = [s[book_id] for s in (svd_scores, item_scores) if book_id in s]
        combined.append((book_id, sum(parts) / len(parts)))

    ranked = sorted(combined, key=lambda x: x[1], reverse=True)[:top_n]
    return _attach_titles(ranked, books)
