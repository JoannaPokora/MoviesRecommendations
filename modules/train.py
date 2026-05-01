#funkcje trainingowe

import pandas as pd
import numpy as np
from sklearn.decomposition import NMF, TruncatedSVD
from .utils import build_rating_matrix
from .utils import build_rating_matrix_sgd
import torch

def train(train_file, model_path, method):
  r"""
  Train the model and store it as an .pkl file
  in the model_path directory.
  """

  Z, user_map, movie_map = build_rating_matrix(train_file, self.method)

  match method:
    case NMF:
      self.W, self.H = train_nmf_model(Z, user_map, movie_map)
    case SVD1:
      self.W, self.H = train_svd1_model(Z, user_map, movie_map)
    case SVD2:
      self.W, self.H = train_svd2_model(Z, user_map, movie_map)
    case SGD:
      self.W, self.H = train_sgd_model(Z, user_map, movie_map)
        
  Z_approx = np.dot(self.W, self.H)

  pickle.dump([Z_approx, user_map, movie_map], open(model_path, 'wb'))
  print("Model saved to", model_path)

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

  ind_optim = np.argmin(diff)

  print("Rank (r):", ind_optim + 4)

  return WH_lst[ind_optim][1], WH_lst[ind_optim][2]


def train_svd1_model(Z):
  svd = TruncatedSVD(n_components=min(Z.shape)-1, random_state=42)
  svd.fit(Z)

  var_expl = np.cumsum(svd.explained_variance_ratio_)
  r = np.argmin(var_expl >= 0.90) + 1
  print("Rank (r):", r)

  svd = TruncatedSVD(n_components=r, random_state=42)
  svd.fit(Z)

  Sigma2 = np.diag(svd.singular_values_)
  VT = svd . components_
  W = svd.transform(Z) / Sigma2
  H = np.dot(Sigma2, VT)
    
  return W, H


def train_svd2_model(train_file):
  svd = TruncatedSVD(n_components=min(Z.shape)-1, random_state=42)
  svd.fit(Z)

  var_expl = np.cumsum(svd.explained_variance_ratio_)
  r = np.argmin(var_expl >= 0.90) + 1
  print("Rank (r):", r)

  svd = TruncatedSVD(n_components=r, random_state=42)
  svd.fit(Z)
  
  Sigma2 = np.diag(svd.singular_values_)
  VT = svd . components_
  W = svd.transform(Z) / Sigma2
  H = np.dot(Sigma2, VT)
    
  return W, H


def train_sgd_model_best_r(train_file, optimizer_name = "adam", r_values=[5, 10, 15, 20, 25], test_size=0.1):
    """
      Reads the ratings CSV file, split data to train and test, builds the rating matrix on training data,
      perform SGD for different values of r, computing RMSE on test data for each r, and returns the best r,
      which minimalize RMSE.

      Parameters:
        - train_file (str): Path to the training CSV file.
        - optimizer_name (str): Word to choose the optimizer (sgd or adam).
        - r_values (list): List of r values to choose best one.
        - test_size (float): Proportion of data to split it to train and test.

      Returns:
        - best_r (int): Value r, which minimalize RMSE on test data.
      """

    df = pd.read_csv(train_file)
    user_map = {uid: i for i, uid in enumerate(sorted(df["userId"].unique()))}
    movie_map = {mid: j for j, mid in enumerate(sorted(df["movieId"].unique()))}

    # podział danych na treningowe i testowe
    df_shuffled = df.sample(frac=1, random_state=42) #losujemy dane
    split_idx = int(len(df_shuffled) * (1 - test_size)) #dzielimy na 0.9 i 0.1 danych
    df_train_split = df_shuffled.iloc[:split_idx] #0.9 danych do trenowania
    df_test_split = df_shuffled.iloc[split_idx:] #0.1 danych do sprawdzania

    # macierz treningową Z (z nan tam, gdzie nie ma ocen lub są w zestawie testowym)
    n_users, n_movies = len(user_map), len(movie_map)
    Z_train = np.full((n_users, n_movies), np.nan, dtype=np.float32)
    for row in df_train_split.itertuples():
        Z_train[user_map[row.userId], movie_map[row.movieId]] = row.rating

    Z_train_torch = torch.from_numpy(Z_train)
    mask_train = ~torch.isnan(Z_train_torch)

    best_r = r_values[0]
    lowest_test_rmse = float('inf')
    best_Z_approx = None

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
            for row in df_test_split.itertuples():
                u_idx, m_idx = user_map[row.userId], movie_map[row.movieId]
                predicted = Z_approx_np[u_idx, m_idx]
                actual = row.rating
                test_errors.append((predicted - actual) ** 2)

            # obliczamy RMSE dla zestawu testowego
            current_test_rmse = np.sqrt(np.mean(test_errors))
            print(f"Walidacyjne RMSE dla r={r}: {current_test_rmse:.4f}")

            # jeśli to r jest lepsze od poprzednich, zapisujemy wynik
            if current_test_rmse < lowest_test_rmse:
                lowest_test_rmse = current_test_rmse
                best_r = r
                best_Z_approx = Z_approx_np

    print(f"Zakończono! Najlepsze r = {best_r} z RMSE = {lowest_test_rmse:.4f}")

    return best_r


def train_sgd_model(train_file, optimizer_name = "adam", r):
    # Używamy wersji SGD, która pozostawia NaN
    Z, user_map, movie_map = build_rating_matrix_sgd(train_file)
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


    Z_approx = torch.matmul(W, H).detach().numpy()

    return Z_approx, user_map, movie_map

