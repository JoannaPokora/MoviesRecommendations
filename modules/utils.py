#uzyteczne funkcje

import pandas as pd
import numpy as np

def build_rating_matrix(train_file):
    """
    Reads a ratings CSV file with columns: userId, movieId, rating.
    Builds and returns the user–movie matrix Z (missing entries set to 0),
    along with mappings from userId to row index and movieId to column index.

    Parameters:
      - train_file (str): Path to the training CSV file.

    Returns:
      - Z (ndarray): Rating matrix of shape (n_users, n_movies).
      - user_map (dict): Mapping from userId to row index.
      - movie_map (dict): Mapping from movieId to column index.
    """
    df = pd.read_csv(train_file)

    # Extract unique users and movies
    unique_users = df["userId"].unique()
    unique_movies = df["movieId"].unique()

    # Create mappings: userId -> row index, movieId -> column index
    user_map = {uid: i for i, uid in enumerate(sorted(unique_users))}
    movie_map = {mid: j for j, mid in enumerate(sorted(unique_movies))}

    n_users = len(user_map)
    n_movies = len(movie_map)

    # Build matrix Z with zeros for missing ratings
    Z = np.full((n_users, n_movies), np.nan, dtype=np.float32)
    for row in df.itertuples():
        u = row.userId
        m = row.movieId
        rating = row.rating
        i = user_map[u]
        j = movie_map[m]
        Z[i, j] = rating

    col_means = np.nanmean(Z, axis=0)
    inds = np.where(np.isnan(Z))
    Z[inds] = np.take(col_means, inds[1])

    # Jeśli po wszystkim nadal są jakieś NaN (filmy bez ani jednej oceny)
    global_mean = np.nanmean(Z)
    Z[np.isnan(Z)] = global_mean
    Z = np.round(Z, 1)

    return Z, user_map, movie_map