"""Series of checks to be performed on dataframes used as inputs of methods fit() and
transform().
"""

from typing import List, Tuple, Union

import numpy as np
import pandas as pd
from scipy.sparse import issparse
from sklearn.utils.validation import _check_y, check_consistent_length, column_or_1d
import polars as pl

def check_X(X: Union[np.generic, np.ndarray, pl.DataFrame, pd.DataFrame]) -> pl.DataFrame:
    """
    Checks if the input is a DataFrame and then creates a copy. This is an important
    step not to accidentally transform the original dataset entered by the user.

    If the input is a numpy array, it converts it to a polars or pandas dataframe. The column
    names are strings representing the column index starting at 0.

    quick_feature was originally designed to work with polars or pandas dataframes. However,
    allowing numpy arrays as input allows 2 things:

    We can use the Scikit-learn tests for transformers provided by the
    `check_estimator` function to test the compatibility of our transformers with
    sklearn functionality.

    quick_feature transformers can be used within a Scikit-learn Pipeline together
    with Scikit-learn transformers like the `SimpleImputer`, which return by default
    Numpy arrays.

    Parameters
    ----------
    X : polars Dataframe or numpy array.
        The input to check and copy or transform.

    Raises
    ------
    TypeError
        If the input is not a polars or pandas dataframe or a numpy array.
    ValueError
        If the input is an empty dataframe.

    Returns
    -------
    X : polars Dataframe.
        A copy of original DataFrame or a converted Numpy array.
    """
    if isinstance(X, pd.DataFrame):
        if not X.columns.is_unique:
            raise ValueError("Input data contains duplicated variable names.")
        data = pl.from_pandas(X)

    elif isinstance(X, (np.generic, np.ndarray)):
        # If input is scalar raise error
        if X.ndim == 0:
            raise ValueError(
                "Expected 2D array, got scalar array instead:\narray={}.\n"
                "Reshape your data either using array.reshape(-1, 1) if "
                "your data has a single feature or array.reshape(1, -1) "
                "if it contains a single sample.".format(X)
            )
        # If input is 1D raise error
        if X.ndim == 1:
            raise ValueError(
                "Expected 2D array, got 1D array instead:\narray={}.\n"
                "Reshape your data either using array.reshape(-1, 1) if "
                "your data has a single feature or array.reshape(1, -1) "
                "if it contains a single sample.".format(X)
            )

        data = pl.DataFrame(X)
        data.columns = [f"x{i}" for i in range(X.shape[1])]
    elif isinstance(X, pl.DataFrame): 
        data = X.clone()
    elif issparse(X):
        raise TypeError("This transformer does not support sparse matrices.")
    else:
        raise TypeError(
            f"X must be a numpy array or polars or pandas dataframe. Got {type(X)} instead."
        )

    if data.__len__() == 0:
        raise ValueError(
            "0 feature(s) (shape=%s) while a minimum of %d is required." % (data.shape, 1)
        )

    return data

def check_y(
    y: Union[np.generic, np.ndarray, pl.Series, pl.DataFrame, List, pd.DataFrame, pd.Series],
    y_numeric: bool = False,
) -> pl.Series:
    """
    Checks that y is a series or a dataframe, or alternatively, if it can be converted
    to a series.

    Parameters
    ----------
    y : pl.Series, pl.DataFrame, np.array, list
        The input to check and copy or transform.

    y_numeric : bool, default=False
        Whether to ensure that y has a numeric type. If dtype of y is object,
        it is converted to float64. Should only be used for regression
        algorithms.

    Returns
    -------
    y: pl.Series
    """


    if  isinstance(y, pl.DataFrame):
        if y.shape[1] == 1:
            y = y.to_series()
        else:
            raise ValueError("Please make sure you are using 1 dimension for Y")
        
    if isinstance(y, pd.DataFrame):
        if y.shape[1] == 1:
            y = y.squeeze(axis=0)
        else:
            raise ValueError("Please make sure you are using 1 dimension for Y")
        
    
    if y is None:
        raise ValueError(
            "requires y to be passed, but the target y is None",
            "Expected array-like (array or non-string sequence), got None",
            "y should be a 1d array",
        )

    elif isinstance(y, pl.Series):
        if y.is_null().any():
            raise ValueError("y contains NaN values.")
        if isinstance(y.dtype, pl.String) == False and not np.isfinite(np.array(y)).all():
            raise ValueError("y contains infinity values.")
        if y_numeric and isinstance(y.dtype, pl.String) :
            y = y.cast(float)
        y_cp = y.clone()

    elif isinstance(y, pd.Series):
        if y.isnull().any():
            raise ValueError("y contains NaN values.")
        if y.dtype != "O" and not np.isfinite(y).all():
            raise ValueError("y contains infinity values.")
        if y_numeric and y.dtype == "O":
            y = y.astype("float")
        y_cp = pl.from_pandas(y)
    else:
        try:
            y_cp = column_or_1d(y)
            y_cp = _check_y(y_cp, multi_output=False, y_numeric=y_numeric)
            y_cp = pl.Series(y_cp).clone()
        except:
            raise ValueError("Please make sure you are using 1 dimension for Y")
    return y_cp



def check_X_y(
    X: Union[np.generic, np.ndarray, pl.DataFrame],
    y: Union[np.generic, np.ndarray, pl.Series, List],
    y_numeric: bool = False,
) -> Tuple[pl.DataFrame, pl.Series]:
    """
    Ensures X and y are compatible polars or pandas dataframe and Series. If both are pandas
    objects, checks that their indexes match. If any is a numpy array, converts to
    pandas object with compatible index.

    This transformer ensures that we can concatenate X and y using `pandas.concat`,
    functionality needed in the encoders.

    Parameters
    ----------
    X: polars or pandas dataframe or numpy ndarray
        The input to check and copy or transform.

    y: pl.Series, np.array, list
        The input to check and copy or transform.

    y_numeric : bool, default=False
        Whether to ensure that y has a numeric type. If dtype of y is object,
        it is converted to float64. Should only be used for regression
        algorithms.

    Raises
    ------
    ValueError: if X and y are pandas objects with inconsistent indexes.
    TypeError: if X is sparse matrix, empty dataframe or not a dataframe.
    TypeError: if y can't be parsed as pandas Series.

    Returns
    -------
    X: polars or pandas dataframe
    y: Pandas Series
    """

    def _check_X_y(X, y):
        X = check_X(X)
        y = check_y(y, y_numeric=y_numeric)
        check_consistent_length(X, y)
        return X, y


    X, y = _check_X_y(X, y)

    return X, y


def _check_X_matches_training_df(X: pl.DataFrame, reference: int) -> None:
    """
    Checks that DataFrame to transform has the same number of columns that the
    DataFrame used with the fit() method.

    Parameters
    ----------
    X : polars or pandas dataframe
        The df to be checked
    reference : int
        The number of columns in the dataframe that was used with the fit() method.

    Raises
    ------
    ValueError
        If the number of columns does not match.

    Returns
    -------
    None
    """
    if X.shape[1] < reference:
        raise ValueError(
            "The number of columns in this dataset is different from the one used to "
            "fit this transformer (when using the fit() method)."
        )

    return None


def _check_contains_na(
    X: pl.DataFrame,
    variables: List[Union[str, int]],
) -> None:
    """
    Checks if DataFrame contains null values in the selected columns.

    Parameters
    ----------
    X : polars or pandas dataframe

    variables : List
        The selected group of variables in which null values will be examined.

    Raises
    ------
    ValueError
        If the variable(s) contain null values.
    """

    if X[variables].null_count().pipe(sum).item() > 0:
        raise ValueError(
            "Some of the variables in the dataset contain NaN. Check and "
            "remove those before using this transformer."
        )


def _check_optional_contains_na(
    X: pl.DataFrame, variables: List[Union[str, int]]
) -> None:
    """
    Checks if DataFrame contains null values in the selected columns.

    Parameters
    ----------
    X : polars or pandas dataframe

    variables : List
        The selected group of variables in which null values will be examined.

    Raises
    ------
    ValueError
        If the variable(s) contain null values.
    """

    if X[variables].null_count().pipe(sum).item() > 0:
        raise ValueError(
            "Some of the variables in the dataset contain NaN. Check and "
            "remove those before using this transformer or set the parameter "
            "`missing_values='ignore'` when initialising this transformer."
        )


def _check_contains_inf(X: pl.DataFrame, variables: List[Union[str, int]]) -> None:
    """
    Checks if DataFrame contains inf values in the selected columns.

    Parameters
    ----------
    X : polars or pandas dataframe
    variables : List
        The selected group of variables in which null values will be examined.

    Raises
    ------
    ValueError
        If the variable(s) contain np.inf values
    """

    if np.isinf(X[variables]).any().any():
        raise ValueError(
            "Some of the variables to transform contain inf values. Check and "
            "remove those before using this transformer."
        )
