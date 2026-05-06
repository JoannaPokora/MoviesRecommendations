#funkcje trainingowe

import pandas as pd
import numpy as np
from sklearn.decomposition import NMF, TruncatedSVD
from .utils import build_rating_matrix
import torch

def train(train_file, method):
  """
  Train the model and store it as an .pkl file
  in the model_path directory.
  """

  Z, user_map, movie_map = build_rating_matrix(train_file, method)

  match method:
    case "NMF":
      W, H = train_nmf_model(Z)
    case "SVD1":
      W, H = train_svd1_model(Z)
    case "SVD2":
      W, H = train_svd2_model(Z)
    case "SGD":
      W, H = train_sgd_model(Z, optimizer_name="adam", r=3)
    case "SGD_s":
      W, H = train_sgd_model(Z, optimizer_name="sgd", r=1)
        
  Z_approx = np.dot(W, H)

  return Z_approx, user_map, movie_map

def train_nmf_model(Z):
  """
  Reads the ratings CSV file, builds the rating matrix using build_rating_matrix,
  performs NMF, and returns the approximated rating matrix along with mappings.

  Parameters:
    - Z (ndarray): Imputed data matrix.

  Returns:
    - W (ndarray): Matrix of size n x r.
    - H (dict): Matrix of size r x d.
  """
 
  r = 3
  WH_lst = []
  rss = []
  while r <= 13:
    model = NMF(n_components=r, init='random', random_state=0, max_iter=1000)
    W = model.fit_transform(Z)
    H = model.components_
    Z_approx = np.dot(W, H)
    WH_lst.append([W, H])
    rss.append(np.sum((Z - Z_approx)**2))
    r += 1

  diff = []
  for i in range(r - 4):
    diff.append(rss[i + 1] - rss[i])

  ind_optim = np.argmin(diff) + 1

  print("Rank (r):", ind_optim + 3)

  return WH_lst[ind_optim][0], WH_lst[ind_optim][1]


def train_svd1_model(Z):
  svd = TruncatedSVD(n_components=min(Z.shape)-1, random_state=42)
  svd.fit(Z)

  var_expl = np.cumsum(svd.explained_variance_ratio_)
  r = np.argmax(var_expl >= 0.9) + 1
  print("Rank (r):", r)

  svd_opt = TruncatedSVD(n_components=r, random_state=42)
  svd_opt.fit(Z)

  Sigma2 = np.diag(svd_opt.singular_values_)
  VT = svd_opt.components_
  W = svd_opt.transform(Z) / svd_opt.singular_values_
  H = np.dot(Sigma2, VT)
    
  return W, H

def train_svd2_model(Z, max_iter=5, tol=1e-3):
    Z_current = Z.copy()

    svd_init = TruncatedSVD(n_components=min(Z.shape) - 1, random_state=42)
    svd_init.fit(Z)

    var_expl = np.cumsum(svd_init.explained_variance_ratio_)
    r = np.argmin(var_expl >= 0.9) + 1
    print("Rank (r):", r)

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

        if diff < tol:
            break

    U = svd.transform(Z_current) / svd.singular_values_
    sqrt_lambda = np.diag(np.sqrt(svd.singular_values_))
    VT = svd.components_

    W = np.dot(U, sqrt_lambda)
    H = np.dot(sqrt_lambda, VT)

    return W, H

def train_sgd_model_best_r(Z, optimizer_name = "adam", r_values=list(range(1,21)), test_size=0.1):
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
    for idx in test_indices:
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


def train_sgd_model(Z, optimizer_name = "adam", r=3):
    r = train_sgd_model_best_r(Z, optimizer_name=optimizer_name)
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

