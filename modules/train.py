#funkcje trainingowe

import pandas as pd
import numpy as np
from sklearn.decomposition import NMF, TruncatedSVD
from .utils import build_rating_matrix, create_cv_folds, evaluate_fold
import torch
from rich.progress import Progress
import warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import root_mean_squared_error

def train(train_file, method):
  """
  Train the model and store it as an .pkl file
  in the model_path directory.
  """

  df = pd.read_csv(train_file)

  match method:
    case "NMF":
      min_r=7
      max_r=40
    case "SVD1":
      min_r=5
      max_r=40
    case "SVD2":
      min_r=2
      max_r=10
    case "SGD":
      min_r=1
      max_r=7
    case "BEST_2":
      min_r = 2
      max_r = 10

  train_fun = globals()[f"train_{method}_model"] # tu określamy funkcję modelu

  folds = create_cv_folds(df, 5, method) # tu tworzymy foldy

  rs = range(min_r, max_r + 1)
  rmse = {}

  with Progress() as p:
    t = p.add_task(description = "initialization", total=len(rs)*len(folds), visible=False)
    for r in rs: # tu sprawdzamy dla każdego r
      p.update(t, description=f"Training with r={r}", refresh=True, visible=True)
      r_rmse = []
      for fold in folds: # tu sprawdzamy po wszystkich foldach
        p.update(t, advance=1)
        Z_train = fold['Z_train']
        train_user_map = fold['user_map']
        train_movie_map = fold['movie_map']
        test_df = fold['test_df']

        W_train, H_train = train_fun(Z_train, r) # tu dopasowujemy model na train

        Z_approx_train = np.dot(W_train, H_train)

        # tu obliczamy i dodajemy rmse dla foldu
        r_rmse.append(evaluate_fold(test_df, train_user_map, train_movie_map, Z_approx_train))
      
      # tu dodajemy srednie rmse dla r
      rmse[r] = np.mean(r_rmse)

  min_rmse = min(rmse.values())
  best_r = list(rmse.keys())[list(rmse.values()).index(min_rmse)]
  print(f"Best r = {best_r} with RMSE = {min_rmse:.4f}")

  Z, user_map, movie_map = build_rating_matrix(df, method)

  if method == "SVD2":
    W, H = train_fun(Z, best_r, max_iter = 1000)
  else:
    W, H = train_fun(Z, best_r)
  Z_approx = np.dot(W, H)

  return Z_approx, user_map, movie_map

def train_NMF_model(Z, r):
  """
  Reads the ratings CSV file, builds the rating matrix using build_rating_matrix,
  performs NMF, and returns the approximated rating matrix along with mappings.

  Parameters:
    - Z (ndarray): Imputed data matrix.

  Returns:
    - W (ndarray): Matrix of size n x r.
    - H (dict): Matrix of size r x d.
  """
  
  with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=ConvergenceWarning)
    model = NMF(n_components=r, init='random', random_state=42, max_iter=1000)
    W = model.fit_transform(Z)
  H = model.components_

  return W, H


def train_SVD1_model(Z, r):
  svd = TruncatedSVD(n_components=r, random_state=42)
  svd.fit(Z)

  Sigma2 = np.diag(svd.singular_values_)
  VT = svd.components_
  W = svd.transform(Z) / svd.singular_values_
  H = np.dot(Sigma2, VT)
    
  return W, H

def train_SVD2_model(Z, r, max_iter=20, tol=1e-6, return_obj = "WH"):
  Z_current = Z.copy()

  # zapamiętujemy, gdzie były oryginalne oceny (większe od 0)
  mask = Z > 0

  prev_rmse = float('inf')

  for i in range(max_iter):
    svd = TruncatedSVD(n_components=r, random_state=42)
    W_iter = svd.fit_transform(Z_current)
    H_iter = svd.components_

    Z_pred = np.dot(W_iter, H_iter)

    # obliczamy zmianę (czy zbiegamy do punktu stałego)
    rmse = root_mean_squared_error(Z_current, Z_pred)
    diff = prev_rmse - rmse
    prev_rmse = rmse

    # Zostawiamy oryginalne oceny, w resztę (braki) wstawiamy przewidywania
    Z_current[~mask] = Z_pred[~mask]

    if diff < tol:
      break

  if return_obj == "Z":
    return Z_current

  U = svd.transform(Z_current) / svd.singular_values_
  sqrt_lambda = np.diag(np.sqrt(svd.singular_values_))
  VT = svd.components_

  W = np.dot(U, sqrt_lambda)
  H = np.dot(sqrt_lambda, VT)

  return W, H

def train_SGD_model(Z, r, optimizer_name = "adam", loss_type = "sq_frob"):
  Z_torch = torch.from_numpy(Z)
  n_users, n_movies = Z_torch.shape

  mask = ~torch.isnan(Z_torch)  # Maska dla znanych ocen

  W = torch.randn((n_users, r), requires_grad=True)
  H = torch.randn((r, n_movies), requires_grad=True)
  if optimizer_name == "adam":
    optimizer = torch.optim.Adam([W, H], lr=0.01)
  elif optimizer_name == "sgd":
    optimizer = torch.optim.SGD([W, H], lr=0.01)
  else:
    raise ValueError("Unsupported optimizer. Choose 'sgd' or 'adam'.")

  for epoch in range(1000):
    optimizer.zero_grad()
    Z_pred = torch.matmul(W, H)
    if loss_type == "sq_frob":
      loss = torch.mean(torch.pow(Z_torch[mask] - Z_pred[mask], 2))
    elif loss_type == "regularization":
      loss = torch.mean(torch.pow(Z_torch[mask] - Z_pred[mask], 2))
    loss.backward()
    optimizer.step()


  return W.detach().numpy(), H.detach().numpy()

def train_BEST_model(Z, svd2_r, nmf_r):
  Z_current = Z.copy()

  # zapamiętujemy, gdzie były oryginalne oceny (większe od 0)
  mask = Z > 0

  prev_rmse = float('inf')

  for i in range(100):
    svd = TruncatedSVD(n_components=svd2_r, random_state=42)
    W_iter = svd.fit_transform(Z_current)
    H_iter = svd.components_

    Z_pred = np.dot(W_iter, H_iter)

    # obliczamy zmianę (czy zbiegamy do punktu stałego)
    rmse = root_mean_squared_error(Z_current, Z_pred)
    diff = prev_rmse - rmse
    prev_rmse = rmse

    # Zostawiamy oryginalne oceny, w resztę (braki) wstawiamy przewidywania
    Z_current[~mask] = Z_pred[~mask]

    if diff < tol:
      break

  return W, H

def train_BEST_2_model(Z_nan, r, max_iter=20, tol=1e-5, alpha=0.8):
  # Zapamiętujemy, gdzie były prawdziwe oceny na podstawie NaN
  mask = ~np.isnan(Z_nan)

  # Przygotowujemy średnie kolumnowe do wygładzania
  col_means = np.nanmean(Z_nan, axis=0)
  global_mean = np.nanmean(Z_nan)
  col_means = np.nan_to_num(col_means, nan=global_mean)

  # Przygotowujemy do SVD2 macierz startową (NaN zastępujemy zerami)
  Z_current = np.nan_to_num(Z_nan, nan=0.0)

  prev_rmse = float('inf')

  for i in range(max_iter):
    svd = TruncatedSVD(n_components=r, random_state=42)
    W_iter = svd.fit_transform(Z_current)
    H_iter = svd.components_

    Z_pred = np.dot(W_iter, H_iter)

    # Obliczamy RMSE dla zbieżności
    rmse = root_mean_squared_error(Z_current, Z_pred)
    diff = abs(prev_rmse - rmse)

    # Dla brakujących wartości (~mask) mieszamy przewidywanie ze średnimi (ZM)
    # np.where(~mask) zwraca indeksy wierszy i kolumn dla braków
    missing_rows, missing_cols = np.where(~mask)
    target_means = col_means[missing_cols] # Pobieramy odpowiednie średnie dla brakujących kolumn

    # Aktualizujemy wartości brakujące
    Z_current[~mask] = alpha * Z_pred[~mask] + (1 - alpha) * target_means

    if diff < tol:
      break

    prev_rmse = rmse

  # Finalny rozkład SVD2
  U = svd.transform(Z_current) / svd.singular_values_
  sqrt_lambda = np.diag(np.sqrt(svd.singular_values_))
  W = np.dot(U, sqrt_lambda)
  H = np.dot(sqrt_lambda, svd.components_)

  return W, H
