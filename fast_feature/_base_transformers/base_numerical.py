""" The base transformer provides functionality that is shared by most transformer
classes. Provides the base functionality within the fit() and transform() methods
shared by most transformers, like checking that input is a df, the size, NA, etc.
"""

import pandas as pd
import polars as pl

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from fast_feature._base_transformers.mixins import GetFeatureNamesOutMixin
from fast_feature.dataframe_checks import (
    _check_contains_inf,
    _check_contains_na,
    _check_X_matches_training_df,
    check_X,
)
from fast_feature.tags import _return_tags
from fast_feature.variable_handling import (
    check_numerical_variables,
    find_numerical_variables,
)


class BaseNumericalTransformer(
    BaseEstimator, TransformerMixin, GetFeatureNamesOutMixin
):
    """Shared set-up procedures across numerical transformers, i.e.,
    variable transformers, discretisers, math combination.
    """

    def fit(self, X: pl.DataFrame) -> pl.DataFrame:
        """
        Checks that input is a dataframe, finds numerical variables, or alternatively
        checks that variables entered by the user are of type numerical.

        Parameters
        ----------
        X : polars or pandas dataframe

        y : Pandas Series, np.array. Default = None
            Parameter is necessary for compatibility with sklearn Pipeline.

        Raises
        ------
        TypeError
            If the input is not a polars or pandas dataframe or a numpy array
            If any of the user provided variables are not numerical
        ValueError
            If there are no numerical variables in the df or the df is empty
            If the variable(s) contain null values

        Returns
        -------
        X : polars or pandas dataframe
            The same dataframe entered as parameter
        """

        # check input dataframe
        X = check_X(X)

        # find or check for numerical variables
        if self.variables is None:
            self.variables_ = find_numerical_variables(X)
        else:
            self.variables_ = check_numerical_variables(X, self.variables)

        # check if dataset contains na or inf
        _check_contains_na(X, self.variables_)
        _check_contains_inf(X, self.variables_)

        # save input features
        self.feature_names_in_ = X.columns

        # save train set shape
        self.n_features_in_ = X.shape[1]

        return X

    def _check_transform_input_and_state(self, X: pl.DataFrame) -> pl.DataFrame:
        """
        Checks that the input is a dataframe and of the same size than the one used
        in the fit() method. Checks absence of NA and Inf.

        Parameters
        ----------
        X : polars or pandas dataframe

        Raises
        ------
        TypeError
            If the input is not a polars or pandas dataframe
        ValueError
            - If the variable(s) contain null values
            - If the df has different number of features than the df used in fit()

        Returns
        -------
        X : polars or pandas dataframe.
            The same dataframe entered by the user.
        """

        # Check method fit has been called
        check_is_fitted(self)

        # check that input is a dataframe
        X = check_X(X)

        # Check if input data contains same number of columns as dataframe used to fit.
        _check_X_matches_training_df(X, self.n_features_in_)

        # check if dataset contains na or inf
        _check_contains_na(X, self.variables_)
        _check_contains_inf(X, self.variables_)

        # reorder variables to match train set
        X = X[self.feature_names_in_]

        return X

    # for the check_estimator tests
    def _more_tags(self):
        tags_dict = _return_tags()
        tags_dict["variables"] = "numerical"
        return tags_dict
