#funkcje trainingowe

import pandas as pd
import numpy as np
from sklearn.decomposition import NMF, TruncatedSVD
from .utils import build_rating_matrix, create_cv_folds, evaluate_fold
import torch
from rich.progress import Progress
import warnings
from sklearn.exceptions import ConvergenceWarning

def train(train_file, method):
  """
  Train the model and store it as an .pkl file
  in the model_path directory.
  """

  df = pd.read_csv(train_file)

  match method:
    case "NMF":
      min_r=5
      max_r=30
      n_folds = 3
    case "SVD1":
      min_r=10
      max_r=40
      n_folds = 5
    case "SVD2":
      min_r=5
      max_r=10
      n_folds = 3
    case "SGD":
      min_r=5
      max_r=30
      n_folds = 3

  train_fun = globals()[f"train_{method}_model"] # tu określamy funkcję modelu

  folds = create_cv_folds(df, n_folds, method) # tu tworzymy foldy

  rs = range(min_r, max_r + 1)
  rmse = {}

  with Progress() as p:
    t = p.add_task(description = "initialization", total=len(rs)*len(folds), visible=False)
    for r in rs: # tu sprawdzamy dla każdego r
      p.update(t, description=f"r = {r}", refresh=True, visible=True)
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

  W, H = train_fun(Z, best_r)
  Z_approx = np.dot(W, H)

  return rmse
  #return Z_approx, user_map, movie_map

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
    model = NMF(n_components=r, init='random', random_state=0, max_iter=1000)
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

def train_SVD2_model(Z, r, max_iter=100, tol=1e-3):
    Z_current = Z.copy()

    # zapamiętujemy, gdzie były oryginalne oceny (większe od 0)
    mask = Z > 0

    for i in range(max_iter):
        svd = TruncatedSVD(n_components=r, random_state=42)
        W_iter = svd.fit_transform(Z_current)
        H_iter = svd.components_

        Z_pred = np.dot(W_iter, H_iter)

        # obliczamy zmianę (czy zbiegamy do punktu stałego)
        diff = np.linalg.norm(Z_current - Z_pred)

        # Zostawiamy oryginalne oceny, w resztę (braki) wstawiamy przewidywania
        Z_current[~mask] = Z_pred[~mask]

        print(diff)

        if diff < tol:
            print("Convergence")
            break

    U = svd.transform(Z_current) / svd.singular_values_
    sqrt_lambda = np.diag(np.sqrt(svd.singular_values_))
    VT = svd.components_

    W = np.dot(U, sqrt_lambda)
    H = np.dot(sqrt_lambda, VT)

    return W, H

def train_SGD_model_best_r(df, Z, optimizer_name = "adam", r_values=list(range(1,50)), test_size=0.1):
    """
      Take Z, split data to train and test, builds the new rating matrix on training data,
      perform SGD for different values of r, computing RMSE on test data for each r, and returns the best r,
      which minimalize RMSE.

      Parameters:
        - Z: Matrix with missing values as NaN.
        - optimizer_name (str): Word to choose the optimizer (sgd or adam).
        - r_values (list[int]): List of r values to choose best one.
        - test_size (float): Proportion of data to split it to train and test.

      Returns:
        - best_r (int): Value r, which minimalize RMSE on test data.

      """

    known_u_idx, known_m_idx = np.where(~np.isnan(Z))
    n_samples = len(known_u_idx)
    # podział danych na treningowe i testowe
    indices = np.arange(n_samples)
    np.random.seed(42)
    np.random.shuffle(indices)
    split_idx = int(n_samples * (1 - test_size))  # dzielimy na 0.9 i 0.1 danych
    train_indices = indices[:split_idx] #0.9 danych do trenowania
    test_indices = indices[split_idx:] #0.1 danych do sprawdzania

    # tworzymy macierz treningową Z
    Z_train = Z.copy()
    for idx in train_indices:
        Z_train[known_u_idx[idx], known_m_idx[idx]] = np.nan

    Z_train_torch = torch.from_numpy(Z_train).float()
    mask_train = ~torch.isnan(Z_train_torch)

    n_users, n_movies = Z.shape
    best_r = r_values[0]
    lowest_test_rmse = float('inf')

    # szukamy best_r, by znaleźć tę z najniższym RMSE
    for r in r_values:
        print(f"--- Testowanie r = {r} ---")

        # inicjalizacja parametrów dla danego r
        W = torch.randn((n_users, r), requires_grad=True)
        H = torch.randn((r, n_movies), requires_grad=True)
        if optimizer_name == "adam":
            optimizer = torch.optim.Adam([W, H], lr=0.01)
        elif optimizer_name == "sgd":
            optimizer = torch.optim.SGD([W, H], lr=0.01)
        else:
            raise ValueError("Unsupported optimizer. Choose 'sgd' or 'adam'.")

        for epoch in range(200):
            optimizer.zero_grad()
            pred = torch.matmul(W, H)
            # liczymy błąd tylko na danych treningowych
            loss = torch.mean(torch.pow(Z_train_torch[mask_train] - pred[mask_train], 2))
            loss.backward()
            optimizer.step()

        # sprawdzamy błąd na danych testowych
        with torch.no_grad():
            Z_approx_np = torch.matmul(W, H).numpy()
            test_errors = []
            for idx in test_indices:
                u = known_u_idx[idx]
                m = known_m_idx[idx]
                actual = Z[u, m]
                predicted = Z_approx_np[u, m]
                test_errors.append((predicted - actual) ** 2)

            # obliczamy RMSE dla zestawu testowego
            current_test_rmse = np.sqrt(np.mean(test_errors))
            print(f"Testowe RMSE dla r={r}: {current_test_rmse:.4f}")

            # jeśli to r jest lepsze od poprzednich, zapisujemy wynik
            if current_test_rmse < lowest_test_rmse:
                lowest_test_rmse = current_test_rmse
                best_r = r
                best_Z_approx = Z_approx_np

    print(f"Zakończono! Najlepsze r = {best_r} z RMSE = {lowest_test_rmse:.4f}")

    return best_r


def train_SGD_model(df, Z, optimizer_name = "adam", r=3):
    r = train_SGD_model_best_r(Z, optimizer_name=optimizer_name)
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
        loss = torch.mean(torch.pow(Z_torch[mask] - Z_pred[mask], 2))
        loss.backward()
        optimizer.step()


    return W.detach().numpy(), H.detach().numpy()

