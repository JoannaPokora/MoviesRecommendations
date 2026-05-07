#funkcje predykcyjne

import pandas as pd
import numpy as np

def predict_rating(row, user_map, movie_map, Z_approx):
        u = row.userId
        m = row.movieId
        if u in user_map.keys() and m in movie_map.keys():
            i = user_map[u]
            j = movie_map[m]
            return Z_approx[i, j]
        else:
            return np.nan

def predict(test_file = None, model_data = None, df = None):
    """
    Reads a test CSV with columns: userId, movieId.
    Uses the stored Z_approx, user_map, movie_map to produce predictions.
    Missing userId/movieId pairs are given average rating of a movie.

    Writes the results to a csv file with columns: 'userId', 'movieId', 'rating'.
    """
    if test_file is not None:
        df = pd.read_csv(test_file)

    Z_approx = model_data["Z_approx"]
    user_map = model_data["user_map"]
    movie_map = model_data["movie_map"]

    df["rating"] = df.apply(predict_rating, axis = 1, args = (user_map, movie_map, Z_approx))

    # imputation
    df["rating"] = (
        df["rating"]
            .fillna(df.groupby("movieId")["rating"].transform("mean")) #grupujemy po filmach i uzupełniamy NaN średniami po nich
            .fillna(df["rating"].mean()) #jeśli nie mamy ocen do danego filmu, bierzemy średnią globalną
            .mul(2).round().div(2) #przybliżenie do 0.5
    )

    return df
