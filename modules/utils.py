#uzyteczne funkcje

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from .predict import predict
from sklearn.metrics import mean_squared_error

def impute_with_col_means(Z):
  """
  Imputes a matrix with column means.

  Parameters:
    - Z (ndarray): Matrix to impute.

  Returns:
    - Z (ndarray): Imputed matrix.
  """

  col_means = np.nanmean(Z, axis=0)
  inds = np.where(np.isnan(Z))
  Z[inds] = np.take(col_means, inds[1])

  return Z

def build_rating_matrix(df, method):
    """
    Takes the train dataframe with columns: userId, movieId, rating.
    Builds and returns the user–movie matrix Z, along with mappings
    from userId to row index and movieId to column index.
    If method is specified as 'NMF' or 'SVD1', missing entries are imputed
    with the mean of movie ratings. For method 'SVD2', they are filled
    with zeros. For other methods, missing entries are nan.

    Parameters:
      - df (ndarray): Train dataframe.
      - method (str): Model name.

    Returns:
      - Z (ndarray): Rating matrix of shape (n_users, n_movies).
      - user_map (dict): Mapping from userId to row index.
      - movie_map (dict): Mapping from movieId to column index.
    """

    # Extract unique users and movies
    unique_users = df["userId"].unique()
    unique_movies = df["movieId"].unique()

    # Create mappings: userId -> row index, movieId -> column index
    user_map = {uid: i for i, uid in enumerate(sorted(unique_users))}
    movie_map = {mid: j for j, mid in enumerate(sorted(unique_movies))}

    n_users = len(user_map)
    n_movies = len(movie_map)

    # Build matrix Z with nan for missing ratings
    Z = np.full((n_users, n_movies), np.nan, dtype=np.float32)
    for row in df.itertuples():
        u = row.userId
        m = row.movieId
        rating = row.rating
        i = user_map[u]
        j = movie_map[m]
        Z[i, j] = rating

    if method in ["NMF", "SVD1"]:
      # impute missing values with means of movies ratings
      Z = impute_with_col_means(Z)
    elif method in ["SVD2"]:
      # impute missing values with zeros
      Z[np.isnan(Z)] = 0

    # round Z entries to nearest 0.5  
    Z = np.round(Z * 2) / 2

    return Z, user_map, movie_map

def create_cv_folds(df, n_splits, method):
  """
  Splits the training dataframe into 'n_splits' folds
  for cross-validation.

  Parameters:
    - df (ndarray): Train dataframe.
    - n_splits (int): Number of folds.
    - method (str): Model name.

  Returns:
    - folds (dict):
  """

  kf = KFold(n_splits, shuffle=True, random_state=42)

  folds = []

  for fold, (train_ind, test_ind) in enumerate(kf.split(df)):
    train_df = df.iloc[train_ind]
    test_df = df.iloc[test_ind]
    
    Z_train, user_map, movie_map = build_rating_matrix(train_df, method)
    
    folds.append({
        'test_df': test_df, 
        'Z_train': Z_train,
        'user_map': user_map,
        'movie_map': movie_map
    })

  return folds

def evaluate_fold(test_df, user_map, movie_map, Z_approx):
  test_preds = test_df[["userId", "movieId"]].copy()
  test_preds = predict(df=test_preds,
                       model_data = {"Z_approx": Z_approx,
                                     "user_map": user_map,
                                     "movie_map": movie_map})
        
  return np.sqrt(mean_squared_error(test_df["rating"], test_preds["rating"]))