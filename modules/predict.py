#funkcje predykcyjne

import pandas as pd
import numpy as np

def predict_rating(row, user_map, movie_map, Z_approx):
  """
  Predicts single rating.

  Parameters:
    - row (pd.Series): Single row of test dataframe.
    - user_map (dict): Mapping from userId to Z_approx row index.
    - user_map (dict): Mapping from movieId to Z_approx column index.
    - Z_approx (ndarray): Model approximation matrix.

  Returns:
    - Rating prediction or nan if the user or movie id
      was not found in the train dataset.
  """

  # get user and movie id from the row
  u = row.userId
  m = row.movieId

  # get prediction from Z approximation
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

  Parameters:
    - test_file (string): Path to the test file with ratings.
    - model_data (dict): Output of train function.
    - df (pd.DataFrame): Test dataframe, used for evaluation in cross-validation.

  Returns:
    - df (pd.DataFrame): Dataframe with predicted ratings.
  """

  # if path given, read test file
  if test_file is not None:
    df = pd.read_csv(test_file)

  # get model data
  Z_approx = model_data["Z_approx"]
  user_map = model_data["user_map"]
  movie_map = model_data["movie_map"]
  
  # predict rating for each row
  df["rating"] = df.apply(predict_rating, axis = 1, args = (user_map, movie_map, Z_approx))

  # imputation
  df["rating"] = (
    df["rating"]
      .fillna(df.groupby("movieId")["rating"].transform("mean")) # group by movies and impute nan with their mean ratings
      .fillna(df["rating"].mean()) # impute with global mean for movies with no ratings
      .mul(2).round().div(2) # round to the nearest 0.5
  )

  return df
