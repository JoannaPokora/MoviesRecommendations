#funkcje trainingowe

import pandas as pd
import numpy as np
from sklearn.decomposition import NMF, TruncatedSVD
from .utils import build_rating_matrix, create_cv_folds, evaluate_fold, impute_with_col_means
import torch
from rich.progress import Progress
import warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import root_mean_squared_error

def train(train_file, method):
  """
  Reads the train file with columns: userId, movieId, rating.
  Perform cross-validation for the specified
  model and return approximated Z matrix,
  users id map and movies id map.

  Parameters:
    - train_file (str): Path to the train file.
    - method(str): Name of the model.

  Returns:
    - Z (ndarray): Matrix of size n x p.
    - user_map (dict): Mapping of users id from original
      data frame to matrix Z.
    - movie_map (dict): Mapping of movies id from original
      data frame to matrix Z.
  """

  # read train file
  df = pd.read_csv(train_file)

  # BEST = SVD2
  if method == "BEST":
    method = "SVD2"

  # set r bounds and lambda if method=SVD2_V2
  match method:
    case "NMF":
      min_r = 7
      max_r = 43
    case "SVD1":
      min_r=5
      max_r=40
    case "SVD2":
      min_r=2
      max_r=10
    case "SGD":
      min_r=1
      max_r=7
    case "SGD_REG":
      min_r=1
      max_r=3
      lam = np.round(np.linspace(0.4, 0.8, 5), 1)
    case "SVD2_V2":
      min_r=2
      max_r=10
    case "NMF_V2":
      min_r=7
      max_r=40

  # set training function
  if method == "SGD_REG":
    train_fun = train_SGD_model
  else:
    train_fun = globals()[f"train_{method}_model"]

  # create folds
  folds = create_cv_folds(df, 5, method)

  # create r range and (r, lambda) pairs if method=SVD2_V2
  rs = range(min_r, max_r + 1)
  if method == "SGD_REG":
    rs = [(r, l) for r in rs for l in lam]

  rmse = {}

  # train loop with different r values and cross-validation
  with Progress() as p:
    # add progress bar
    t = p.add_task(description = "initialization", total=len(rs)*len(folds), visible=False)

    # train with single r
    for r in rs:
      # update progress bar
      if method == "SGD_REG":
        p.update(t, description=f"Training with r={r[0]} and lambda={r[1]}", refresh=True, visible=True)
      else:
        p.update(t, description=f"Training with r={r}", refresh=True, visible=True)

      r_rmse = []

      # cross-validation
      for fold in folds:
        p.update(t, advance=1)
        Z_train = fold['Z_train']
        train_user_map = fold['user_map']
        train_movie_map = fold['movie_map']
        test_df = fold['test_df']

        # fit model
        if method == "SGD_REG":
          W_train, H_train = train_fun(Z_train, r[0], loss_type = "reg", lam = r[1])
        else:
          W_train, H_train = train_fun(Z_train, r)

        Z_approx_train = np.dot(W_train, H_train)

        # calculate RMSE on test fold
        r_rmse.append(evaluate_fold(test_df, train_user_map, train_movie_map, Z_approx_train))
      
      # add mean RMSE for single r
      rmse[r] = np.mean(r_rmse)

  # get the best r or (r, lambda) for SVD2_V2
  min_rmse = min(rmse.values())
  best_r = list(rmse.keys())[list(rmse.values()).index(min_rmse)]
  if method == "SGD_REG":
    print(f"Best r={best_r[0]} and lambda={best_r[1]} with RMSE = {min_rmse:.4f}")
  else:
    print(f"Best r={best_r} with RMSE = {min_rmse:.4f}")

  # get full data matrix
  Z, user_map, movie_map = build_rating_matrix(df, method)

  # train on full data matrix
  if method in ("SVD2", "SVD2_V2"):
    W, H = train_fun(Z, best_r, max_iter = 1000)
  elif method == "SGD_REG":
    W, H = train_fun(Z, best_r[0], loss_type = "reg", lam = best_r[1])
  else:
    W, H = train_fun(Z, best_r)

  # get approximation
  Z_approx = np.dot(W, H)

  return Z_approx, user_map, movie_map

def train_NMF_model(Z, r):
  """
  Performs NMF approximation.

  Parameters:
    - Z (ndarray): Imputed data matrix.
    - r (int): Rank parameter.

  Returns:
    - W (ndarray): Matrix of size n x r.
    - H (dict): Matrix of size r x p.
  """
  
  # ido not print warnings
  with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=ConvergenceWarning)
    model = NMF(n_components=r, init='nndsvd', random_state=42, max_iter=1000)
    W = model.fit_transform(Z)
  H = model.components_

  return W, H


def train_SVD1_model(Z, r):
  """
  Performs truncated SVD.

  Parameters:
    - Z (ndarray): Imputed data matrix.
    - r (int): Rank parameter.

  Returns:
    - W (ndarray): Matrix of size n x r.
    - H (dict): Matrix of size r x p.
  """
    
  svd = TruncatedSVD(n_components=r, random_state=42)
  W = svd.fit_transform(Z)
  H = svd.components_
    
  return W, H

def train_SVD2_model(Z, r, max_iter=20, tol=1e-6, mask = None):
  """
  Performs SVD2 - iterative truncated SVD with imputation.

  Parameters:
    - Z (ndarray): Data matrix with zeros (or imputed for SVD_V2).
    - r (int): Rank parameter.
    - max_iter (int): Maximum number of iterations.
    - tol (float): Stopping criterion value.
    - mask (ndarray): Positions of initially missing entries,
      used with SVD_V2.

  Returns:
    - W (ndarray): Matrix of size n x r.
    - H (dict): Matrix of size r x p.
  """

  Z_current = Z.copy()

  # remember the positions of known ratings
  if mask is None:
    mask = Z > 0

  # set initial RMSE as infinity
  prev_rmse = float('inf')

  # perform SVD2 loop
  for i in range(max_iter):
    svd = TruncatedSVD(n_components=r, random_state=42)
    W_iter = svd.fit_transform(Z_current)
    H_iter = svd.components_

    Z_pred = np.dot(W_iter, H_iter)

    # calculate difference between RMSE values
    rmse = root_mean_squared_error(Z_current[mask], Z_pred[mask])
    diff = prev_rmse - rmse
    prev_rmse = rmse

    # impute missing values
    Z_current[~mask] = Z_pred[~mask]

    # if difference small - break for
    if diff < tol:
      break

  return W_iter, H_iter

def train_SGD_model(Z, r, optimizer_name = "adam", loss_type = "sq_frob", lam = 0.5):
  """
  Performs SGD to find the approximation of Z.

  Parameters:
    - Z (ndarray): Imputed data matrix.
    - r (int): Rank parameter.
    - optimizer_name (str): Specifies which optimizer to use.
      Can be 'adam' (default) or 'sgd'.
    - loss_type (str): Specifies the loss type.
      Can be 'sq_frob' (default) or 'reg',
      for loss with regularization.
    - lam (float): The lambda parameter for
      the regularized loss.

  Returns:
    - W (ndarray): Matrix of size n x r.
    - H (dict): Matrix of size r x p.
  """

  torch.manual_seed(42)
  
  # initialize torch matrix
  Z_torch = torch.from_numpy(Z)
  n_users, n_movies = Z_torch.shape

  # remember the positions of known ratings
  mask = ~torch.isnan(Z_torch)

  # create W and H torch matrices
  W = torch.randn((n_users, r), requires_grad=True)
  H = torch.randn((r, n_movies), requires_grad=True)

  # set optimizer
  if optimizer_name == "adam":
    optimizer = torch.optim.Adam([W, H], lr=0.01)
  elif optimizer_name == "sgd":
    optimizer = torch.optim.SGD([W, H], lr=0.01)
  else:
    raise ValueError("Unsupported optimizer. Choose 'sgd' or 'adam'.")

  # perform SGD
  for epoch in range(1000):
    optimizer.zero_grad()

    # predict Z
    Z_pred = torch.matmul(W, H)

    # calculate loss
    sum_sq_diff = torch.sum(torch.pow(Z_torch[mask] - Z_pred[mask], 2))
    if loss_type == "reg":
      reg = lam * (torch.sum(torch.pow(W, 2)) + torch.sum(torch.pow(H, 2)))
      loss = sum_sq_diff + reg
    else:
      loss = sum_sq_diff

    loss.backward()
    optimizer.step()

  return W.detach().numpy(), H.detach().numpy()

def train_SVD2_V2_model(Z_nan, r, max_iter=20, tol=1e-6):
  """
  Performs SVD2 with initial mean imputation.

  Parameters:
    - Z (ndarray): Imputed data matrix.
    - r (int): Rank parameter.
    - max_iter (int): Maximum number of iterations.
    - tol (float): Stopping criterion value.

  Returns:
    - W (ndarray): Matrix of size n x r.
    - H (dict): Matrix of size r x p.
  """
  # remember the positions of known ratings
  mask = ~np.isnan(Z_nan)

  # impute matrix with column means
  Z = impute_with_col_means(Z_nan)

  # perform SVD2
  W, H = train_SVD2_model(Z, r, max_iter=max_iter, tol=tol, mask = mask)

  return W, H

def train_NMF_V2_model(Z_nan, r):
  """
  Performs "double" NMF approximation - first run imputes
  missing values and second run performs NMF with rounded r/2.

  Parameters:
    - Z_nan (ndarray): Data matrix with missing entries
      filled with nan.
    - r (int): Rank parameter.

  Returns:
    - W (ndarray): Matrix of size n x r.
    - H (dict): Matrix of size r x p.
  """

  # remember the positions of known ratings
  mask = ~np.isnan(Z_nan)

  # impute matrix with column means
  Z = impute_with_col_means(Z_nan.copy())

  # perform NMF first time
  W, H = train_NMF_model(Z, r)

  # impute missing ratings
  Z_prim = np.dot(W, H)
  Z_prim[mask] = Z[mask]

  # perform NMF second time
  W, H = train_NMF_model(Z_prim, round(r/2))

  return W, H
