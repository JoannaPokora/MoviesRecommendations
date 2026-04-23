#funkcje trainingowe

import pandas as pd
import numpy as np
from sklearn.decomposition import NMF
from sklearn.metrics import explained_variance_score
from .utils import build_rating_matrix

def train_nmf_model(train_file):
  """
  Reads the ratings CSV file, builds the rating matrix using build_rating_matrix,
  performs NMF, and returns the approximated rating matrix along with mappings.

  Parameters:
    - train_file (str): Path to the training CSV file.

  Returns:
    - Z_approx (ndarray): Approximated rating matrix from NMF.
    - user_map (dict): Mapping from userId to row index.
    - movie_map (dict): Mapping from movieId to column index.
  """
  Z, user_map, movie_map = build_rating_matrix(train_file)

  n = 1
  error = 500
  Z_approx = 0
  while n < 50 and error > 250:
    model = NMF(n_components=n, init='random', random_state=0, max_iter=1000)
    W = model.fit_transform(Z)
    H = model.components_
    Z_approx = np.dot(W, H)
    error = np.linalg.norm(Z - Z_approx, 'fro')
    print(error)
    print(n)
    n += 1
    
  return Z_approx, user_map, movie_map