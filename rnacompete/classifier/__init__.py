"""
rnacompete/classifier/__init__.py

Enables the loading of classifier parameters.
"""

from importlib import resources
import pickle
from typing import Tuple

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression


def load_classifier() -> Tuple[StandardScaler, LinearRegression]:
    """
    Return the classifier parameters.

    Returns
    -------
    ss : sklearn.preprocessing.StandardScaler
        Standard scaler model.
    lr : sklearn.linear_model.LinearRegression
        Linear regression model.
    """
    with resources.open_binary(__package__, 'SS_model.sav') as f:
        ss = pickle.load(f)
    with resources.open_binary(__package__, 'LR_model.sav') as f:
        lr = pickle.load(f)

    return ss, lr
