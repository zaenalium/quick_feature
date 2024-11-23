"""Dataframe used as input by many estimator checks."""

from typing import Tuple

import pandas as pd
from sklearn.datasets import make_classification
import polars as pl

def test_df(
    categorical: bool = False, datetime: bool = False
) -> Tuple[pl.DataFrame, pl.Series]:
    """
    Creates a dataframe that contains only numerical features, or additionally,
    categorical and datetime features.

    Parameters
    ----------
    categorical: bool, default=False
        Whether to add 2 additional categorical features.

    datetime: bool, default=False
        Whether to add one additional datetime feature.

    Returns
    -------
    X: pl.DataFrame
        A polars or pandas dataframe.
    """
    X, y = make_classification(
        n_samples=1000,
        n_features=12,
        n_redundant=4,
        n_clusters_per_class=1,
        weights=[0.50],
        class_sep=2,
        random_state=1,
    )

    # transform arrays into pandas df and series
    colnames = [f"var_{i}" for i in range(12)]
    X = pl.DataFrame(X, schema=colnames)
    y = pl.Series(y)

    if categorical is True:
        X = X.with_columns(cat_var1 = ["A"] * 1000)
        X = X.with_columns(cat_var2 = ["B"] * 1000) 

    if datetime is True:
        X = X.with_columns(date1 = pd.date_range("2020-02-24", periods=1000, freq="min").to_list())
        X = X.with_columns(date2 = pd.date_range("2021-09-29", periods=1000, freq="h").to_list())
    return X, y
