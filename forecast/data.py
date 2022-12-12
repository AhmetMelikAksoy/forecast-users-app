from pathlib import Path
from argparse import Namespace
from typing import List

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from statsmodels.tsa.deterministic import DeterministicProcess

from config import config
from forecast import utils

import warnings

warnings.filterwarnings("ignore")


def elt_data() -> pd.DataFrame:
    """Load and prepare the data from data directory."""
    df = pd.read_csv(Path(config.DATA_DIR, config.DATA_PATH), sep=";")
    df["time"] = pd.to_datetime(df["time"])
    df.set_index("time", inplace=True)
    df = df.asfreq("H")

    return df


def create_inputs_for_trend_model(
    index: pd.DatetimeIndex,
    args_fp: str) -> pd.DataFrame:
    """Creates features for the linear trend model."""

    args = Namespace(**utils.load_dict(filepath=Path(config.CONFIG_DIR, args_fp)))
    dp = DeterministicProcess(
        index= index,
        constant=args.constant,
        order=args.order,
        drop=args.drop,
        fourier=args.fourier
    )
    
    return dp.in_sample()


def get_data_splits(X:pd.DataFrame, y:pd.DataFrame, train_size:float) -> pd.DataFrame:
    idx_train, idx_test = train_test_split(X.index, train_size=train_size, shuffle=False)

    X_train, y_train = X.loc[idx_train, :], y.loc[idx_train, :]
    X_test, y_test = X.loc[idx_test, :], y.loc[idx_test, :]

    return X_train, y_train, X_test, y_test


class LagFeatureTransformer(BaseEstimator, TransformerMixin):
    """Class for adding lag feature creation to the sklearn pipeline. It creates lag and future step features."""
    
    def __init__(self, variables: List) -> None:
        
        if not isinstance(variables, list):
            raise ValueError("variables need to be provided as a list")

        self.variables = variables

    def fit(self, X, y=None):
        return self

    def transform(
        self, 
        X: pd.DataFrame,
        lag_steps: int = 0,
        feature_steps: int = 0,
        drop_na: bool = True) -> pd.DataFrame:

        X = X.copy()

        for feature in self.variables:
            for i in range(1, lag_steps+1):
                X[f"{feature}_(t-{i})"] = X[feature].shift(i)
            for i in range(1, feature_steps+1):
                X[f"{feature}_(t+{i})"] = X[feature].shift(-i)
        
        if drop_na:
            X = X.dropna()

        return X


class BinaryEncoder(BaseEstimator, TransformerMixin):
    """Transforming numeric values to binary encoded series."""

    def __init__(self, variables: List) -> None:
        if not isinstance(variables, list):
            raise ValueError("variables need to be provided as a list")

        self.variables = variables


    def fit(self, X, y=None):
        return self


    def transform(self, X, threshold: int=0):
        X = X.copy()

        for feature in self.variables:
            X[feature] = np.where(X[feature] > threshold, 1, 0)

        return X