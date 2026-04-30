#funkcje predykcyjne

import pandas as pd
import numpy as np
from .utils import build_rating_matrix

def predict(test_file, model_data):
    """
    Reads a test CSV with columns: userId, movieId.
    Uses the stored Z_approx, user_map, movie_map to produce predictions.
    Missing userId/movieId combos produce a default rating (e.g., 0 or average).

    Returns a list of dicts with keys: 'userId', 'movieId', 'rating'.
    """
    df = pd.read_csv(test_file)

    Z_approx, user_map, movie_map = model_data

    def predict_rating(row):
        u = row.userId
        m = row.movieId
        if u in user_map and m in movie_map:
            i = user_map[u]
            j = movie_map[m]
            return Z_approx[i, j]
        else:
            return np.nan
        
    df["rating"] = df.apply(predict_rating, axis = 1)

    # imputation
    df["rating"] = df["rating"].fillna(df.groupby("movieId")["rating"].transform("mean")).mul(2).round().div(2)

  #  predictions = []
  #  for row in df.itertuples():
  #      predictions.append({
  #          "userId": row.userId,
  #          "movieId": row.movieId,
  #          "rating": row.rating
  #      })

    return df