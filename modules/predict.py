#funkcje predykcyjne

import pandas as pd
import numpy as np
from .utils import build_rating_matrix

def predict(model_data, test_file, output_file):
    """
    Reads a test CSV with columns: userId, movieId.
    Uses the stored Z_approx, user_map, movie_map to produce predictions.
    Missing userId/movieId pairs are given average rating of a movie.

    Writes the results to a csv file with columns: 'userId', 'movieId', 'rating'.
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
    df["rating"] = (
        df["rating"]
            .fillna(df.groupby("movieId")["rating"].transform("mean")) #grupujemy po filmach i uzupełniamy NaN średniami po nich
            .fillna(df["rating"].mean()) #jeśli nie mamy ocen do danego filmu, bierzemy średnią globalną
            .mul(2).round().div(2) #przybliżenie do 0.5
    )

    df.to_csv(output_file, index=False)

    return df
