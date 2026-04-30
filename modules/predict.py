#funkcje predykcyjne

import pandas as pd
import numpy as np
from .utils import build_rating_matrix

def predict(test_file, model_data):
    """
    Reads a test CSV with columns: userId, movieId.
    Uses the stored Z_approx, user_map, movie_map to produce predictions.
    Missing userId/movieId combos produce a default rating (e.g., 0 or average).

    Returns a list of dicts with keys: 'userId', 'movieId', 'rating'.
    """
    df = pd.read_csv(test_file)

    Z_approx, user_map, movie_map = model_data

    def predict_rating(row):
        u = row.userId
        m = row.movieId
        if u in user_map and m in movie_map:
            i = user_map[u]
            j = movie_map[m]
            return Z_approx[i, j]
        else:
            return np.nan

    df["rating"] = df.apply(predict_rating, axis = 1)

    # imputation
    df["rating"] = df["rating"].fillna(df.groupby("movieId")["rating"].transform("mean")).mul(2).round().div(2)

  #  predictions = []
  #  for row in df.itertuples():
  #      predictions.append({
  #          "userId": row.userId,
  #          "movieId": row.movieId,
  #          "rating": row.rating
  #      })

    return df


def predict_sgd(test_file, model_data):
    """
    Reads a test CSV with columns: userId, movieId.
    Uses the stored Z_approx, user_map, movie_map to produce predictions.

    Returns a list of dicts with keys: 'userId', 'movieId', 'rating'.
    """
    df = pd.read_csv(test_file)
    Z_approx, user_map, movie_map = model_data
    predictions = []

    # Obliczamy globalną średnią z Z_approx na wypadek napotkania nowych użytkowników/filmów
    global_mean = np.nanmean(Z_approx)

    for row in df.itertuples():
        uid = row.userId
        mid = row.movieId
        if uid in user_map and mid in movie_map:
            u_idx = user_map[uid]
            m_idx = movie_map[mid]
            pred = Z_approx[u_idx, m_idx]
            pred = round(pred * 2) / 2
            # pilnujemy zakresu 0 - 5.0
            pred = max(0, min(5, pred))
        else:
            # Jeśli user/film jest nowy (nie było go w train), dajemy średnią
            pred = round(global_mean * 2) / 2

        predictions.append(pred)

    return predictions


def predict_sgd_wersja2(test_file, model_data):
    """
    Przewiduje oceny, stosując hierarchię:
    1. Model (jeśli znamy usera i film)
    2. Średnia po filmie (jeśli film znamy, ale user nie)
    3. Średnia po użytkowniku (jeśli usera znamy, ale filmu nie)
    4. Średnia globalna (jeśli obu nie znamy)
    """
    test_df = pd.read_csv(test_file)
    Z_approx, user_map, movie_map = model_data
    predictions = []

    # Przygotowujemy średnie pomocnicze (na surowych danych z Z_approx)
    # axis=0 to średnie po kolumnach (filmach), axis=1 to średnie po wierszach (userach)
    movie_means = np.nanmean(Z_approx, axis=0)
    user_means = np.nanmean(Z_approx, axis=1)
    global_mean = np.nanmean(Z_approx)

    for row in test_df.itertuples():
        uid = row.userId
        mid = row.movieId

        # SCENARIUSZ A: Znamy obu (używamy wyuczonego modelu)
        if uid in user_map and mid in movie_map:
            u_idx = user_map[uid]
            m_idx = movie_map[mid]
            pred = Z_approx[u_idx, m_idx]

        # SCENARIUSZ B: Znamy film, ale nie znamy użytkownika
        elif mid in movie_map:
            m_idx = movie_map[mid]
            pred = movie_means[m_idx]  # Średnia ocena tego konkretnego filmu

        # SCENARIUSZ C: Znamy użytkownika, ale nie znamy filmu
        elif uid in user_map:
            u_idx = user_map[uid]
            pred = user_means[u_idx]  # Średnia jak ten użytkownik zwykle ocenia

        # SCENARIUSZ D: Całkowicie nowa para
        else:
            pred = global_mean


        pred = round(pred * 2) / 2
        pred = max(0.5, min(5, pred))

        predictions.append(pred)

    return predictions