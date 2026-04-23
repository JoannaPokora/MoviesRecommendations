#funkcje trainingowe

import pandas as pd
import numpy as np
from sklearn.decomposition import NMF, TruncatedSVD
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
  error = 300
  Z_approx = 0
  while n < 15 and error > 250:
    model = NMF(n_components=n, init='random', random_state=0, max_iter=1000)
    W = model.fit_transform(Z)
    H = model.components_
    Z_approx = np.dot(W, H)
    error = np.linalg.norm(Z - Z_approx, 'fro')
    print(error)
    print(n)
    n += 1
    
  return Z_approx, user_map, movie_map


def train_svd1_model(train_file):
  Z, user_map, movie_map = build_rating_matrix(train_file)

  svd_100_comp = TruncatedSVD(n_components=100, random_state=42)
  svd_100_comp.fit(Z)
  Sigma2 = np.diag(svd_100_comp.singular_values_)
  VT = svd_100_comp.components_
  W = svd_100_comp.transform(Z) / svd_100_comp.singular_values_
  H = np.dot(Sigma2, VT)
  Z_approx = np.dot(W, H)
  cum_var_explained = np.cumsum(svd_100_comp.explained_variance_)

  if(any(cum_var_explained >= 90)):
    n = np.argwhere(cum_var_explained >= 2)[0, 0]
    if(n == 100):
      return Z_approx, user_map, movie_map
  else:
    n = 101
  
  var_explained = 0
  Z_approx = 0
  while var_explained < 90:
    svd = TruncatedSVD(n_components=n, random_state=42)
    svd.fit(Z)
    Sigma2 = np.diag(svd.singular_values_)
    VT = svd.components_
    W = svd.transform(Z) / svd.singular_values_
    H = np.dot(Sigma2, VT)
    Z_approx = np.dot(W, H)
    error = np.sum(svd.explained_variance_)
    print(error)
    print(n)
    n += 1
    
  return Z_approx, user_map, movie_map


def train_svd2_model(train_file):
  Z, user_map, movie_map = build_rating_matrix(train_file)

  n = 1
  error = 300
  Z_approx = 0
  while n <= 15 and error > 250:
    svd = TruncatedSVD(n_components=n, random_state=42)
    svd.fit(Z)
    Sigma2 = np.diag(svd.singular_values_)
    VT = svd.components_
    W = svd.transform(Z) / svd.singular_values_
    H = np.dot(Sigma2, VT)
    Z_approx = np.dot(W, H)
    error = np.linalg.norm(Z - Z_approx, 'fro')
    print(error)
    print(n)
    n += 1
    
  return Z_approx, user_map, movie_map