"""
Book Recommendation System -- CLI demo

python main.py --user 7 --method svd --top_n 10
    python main.py --evaluate      # prints RMSE of the SVD model
"""

import argparse
import pandas as pd

from src.data_loader import load_data, build_matrix
from src.recommender import (
    user_based_recommend,
    item_based_recommend,
    svd_recommend,
    hybrid_recommend,
)
from src.evaluate import evaluate_svd

METHODS = {
    "user": user_based_recommend,
    "item": item_based_recommend,
    "svd": svd_recommend,
    "hybrid": hybrid_recommend,
}


def main():
    parser = argparse.ArgumentParser(description="Book Recommendation System")
    parser.add_argument("--user", type=int, default=1, help="user_id to recommend for")
    parser.add_argument("--top_n", type=int, default=5, help="number of recommendations")
    parser.add_argument(
        "--method", choices=list(METHODS) + ["all"], default="all",
        help="which algorithm to use",
    )
    parser.add_argument("--evaluate", action="store_true", help="print SVD RMSE and exit")
    args = parser.parse_args()

    books, ratings = load_data()
    matrix = build_matrix(ratings, books)

    if args.evaluate:
        rmse = evaluate_svd(ratings, books)
        print(f"SVD model RMSE on held-out test ratings: {rmse:.3f}")
        return

    if args.user not in matrix.index:
        print(f"user_id {args.user} not found. Valid range: 1-{matrix.index.max()}")
        return

    print(f"\nBooks user {args.user} already rated:")
    print(matrix.loc[args.user].dropna())

    methods_to_run = METHODS if args.method == "all" else {args.method: METHODS[args.method]}
    for name, fn in methods_to_run.items():
        print(f"\n--- {name.upper()}-BASED RECOMMENDATIONS ---")
        results = fn(matrix, args.user, books, top_n=args.top_n)
        print(pd.DataFrame(results))


if __name__ == "__main__":
    main()


