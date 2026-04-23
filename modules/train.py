#funkcje trainingowe

import pandas as pd
import numpy as np
from sklearn.decomposition import NMF
from .helper_functions import build_rating_matrix


def train_nmf_model(train_file, n_components=5):
    """
    Reads the ratings CSV file, builds the rating matrix using build_rating_matrix,
    performs NMF, and returns the approximated rating matrix along with mappings.

    Parameters:
      - train_file (str): Path to the training CSV file.
      - n_components (int): Rank for the NMF decomposition.

    Returns:
      - Z_approx (ndarray): Approximated rating matrix from NMF.
      - user_map (dict): Mapping from userId to row index.
      - movie_map (dict): Mapping from movieId to column index.
    """
    Z, user_map, movie_map = build_rating_matrix(train_file)

    model = NMF(n_components=n_components, init='random', random_state=0, max_iter=1000)
    W = model.fit_transform(Z)
    H = model.components_
    Z_approx = np.dot(W, H)

    return Z_approx, user_map, movie_map