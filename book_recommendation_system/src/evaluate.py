"""RMSE evaluation for the SVD recommender on a held-out test split."""

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd


def evaluate_svd(ratings, books, n_components=8, test_size=0.2, seed=42):
    from .data_loader import build_matrix

    train, test = train_test_split(ratings, test_size=test_size, random_state=seed)

    train_matrix = build_matrix(train, books)
    global_mean = train_matrix.stack().mean()
    filled = train_matrix.fillna(global_mean)

    nc = min(n_components, min(filled.shape) - 1)
    svd = TruncatedSVD(n_components=nc, random_state=seed)
    user_factors = svd.fit_transform(filled)
    item_factors = svd.components_
    predicted = pd.DataFrame(
        user_factors @ item_factors, index=train_matrix.index, columns=train_matrix.columns
    )

    y_true, y_pred = [], []
    for _, row in test.iterrows():
        if row.user_id in predicted.index and row.book_id in predicted.columns:
            y_true.append(row.rating)
            y_pred.append(np.clip(predicted.loc[row.user_id, row.book_id], 1, 5))

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return rmse
