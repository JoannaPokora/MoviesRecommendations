#funkcje predykcyjne

import pandas as pd
import numpy as np


def predict_nmf(test_file, model_data):
    """
    Reads a test CSV with columns: userId, movieId.
    Uses the stored Z_approx, user_map, movie_map to produce predictions.
    Missing userId/movieId combos produce a default rating (e.g., 0 or average).

    Returns a list of dicts with keys: 'userId', 'movieId', 'rating'.
    """
    df = pd.read_csv(test_file)

    Z_approx = model_data["Z_approx"]
    user_map = model_data["user_map"]
    movie_map = model_data["movie_map"]

    predictions = []
    for row in df.itertuples():
        u = row.userId
        m = row.movieId
        if u in user_map and m in movie_map:
            i = user_map[u]
            j = movie_map[m]
            rating = Z_approx[i, j]
        else:
            # If user or movie not seen in training, default to 0 (or any strategy)
            rating = 0
        # Round rating to nearest 0.5 increment
        rating_rounded = round(rating * 2) / 2

        predictions.append({
            "userId": u,
            "movieId": m,
            "rating": rating_rounded
        })
    return predictions