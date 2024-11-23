import warnings
from typing import List, Union

import polars as pl
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from fast_feature._base_transformers.mixins import GetFeatureNamesOutMixin
from fast_feature._check_init_parameters.check_variables import (
    _check_variables_input_value,
)
from fast_feature._docstrings.init_parameters.all_trasnformers import (
    _missing_values_docstring,
    _variables_categorical_docstring,
)
from fast_feature._docstrings.init_parameters.encoders import _ignore_format_docstring
from fast_feature._docstrings.substitute import Substitution
from fast_feature.dataframe_checks import (
    _check_optional_contains_na,
    _check_X_matches_training_df,
    check_X,
)
from fast_feature.tags import _return_tags
from fast_feature.variable_handling import (
    check_all_variables,
    check_categorical_variables,
    find_all_variables,
    find_categorical_variables,
)


@Substitution(
    ignore_format=_ignore_format_docstring,
    variables=_variables_categorical_docstring,
)
class CategoricalInitMixin:
    """Shared initialization parameters across transformers. Sets and checks init
    parameters.

    Parameters
    ----------
    {variables}.

    {ignore_format}
    """

    def __init__(
        self,
        variables: Union[None, int, str, List[Union[str, int]]] = None,
        ignore_format: bool = False,
    ) -> None:

        if not isinstance(ignore_format, bool):
            raise ValueError(
                "ignore_format takes only booleans True and False. "
                f"Got {ignore_format} instead."
            )

        self.variables = _check_variables_input_value(variables)
        self.ignore_format = ignore_format


@Substitution(
    missing_values=_missing_values_docstring,
    ignore_format=_ignore_format_docstring,
    variables=_variables_categorical_docstring,
)
class CategoricalInitMixinNA:
    """Shared initialization parameters across transformers. Sets and checks init
    parameters.

    Parameters
    ----------
    {variables}.

    {missing_values}

    {ignore_format}
    """

    def __init__(
        self,
        variables: Union[None, int, str, List[Union[str, int]]] = None,
        missing_values: str = "raise",
        ignore_format: bool = False,
    ) -> None:

        if missing_values not in ["raise", "ignore"]:
            raise ValueError(
                "missing_values takes only values 'raise' or 'ignore'. "
                f"Got {missing_values} instead."
            )

        if not isinstance(ignore_format, bool):
            raise ValueError(
                "ignore_format takes only booleans True and False. "
                f"Got {ignore_format} instead."
            )

        self.variables = _check_variables_input_value(variables)
        self.ignore_format = ignore_format
        self.missing_values = missing_values


class CategoricalMethodsMixin(BaseEstimator, TransformerMixin, GetFeatureNamesOutMixin):
    """Shared methods across categorical transformers.

    - BaseEstimator brings methods get_params() and set_params().
    - TransformerMixin brings method fit_transform()
    - GetFeatureNamesOutMixin brings method get_feature_names_out().
    """

    def _check_na(self, X: pl.DataFrame, variables):
        if self.missing_values == "raise":
            _check_optional_contains_na(X, variables)

    def _check_or_select_variables(self, X: pl.DataFrame):
        """
        Finds categorical variables, or alternatively checks that the variables
        entered by the user are of type object (categorical).
        Checks absence of NA.

        Parameters
        ----------
        X: polars or pandas dataframe

        Raises
        ------
        TypeError
            If any user provided variable is not categorical
        ValueError
            If there are no categorical variables in the df or the df is empty
            If the variable(s) contain null values
        """
        # select variables to encode
        if self.ignore_format is True:
            if self.variables is None:
                variables_ = find_all_variables(X)
            else:
                variables_ = check_all_variables(X, self.variables)
        else:
            if self.variables is None:
                variables_ = find_categorical_variables(X)
            else:
                variables_ = check_categorical_variables(X, self.variables)

        return variables_

    def _get_feature_names_in(self, X: pl.DataFrame):
        """
        Returns attributes `featrure_names_in_` and `n_feature_names_in_`, which are
        standard for all transformers in the library.
        """
        # save input features
        self.feature_names_in_ = X.columns

        # save train set shape
        self.n_features_in_ = X.shape[1]

    def _check_transform_input_and_state(self, X: pl.DataFrame) -> pl.DataFrame:
        """
        Checks that the input is a dataframe and of the same size than the one used
        in the fit method. Checks absence of NA.

        Parameters
        ----------
        X: polars or pandas dataframe

        Raises
        ------
        TypeError
            If the input is not a polars or pandas dataframe
        ValueError
            - If the variable(s) contain null values.
            - If the df has different number of features than the df used in fit()

        Returns
        -------
        X: polars or pandas dataframe
            The same dataframe entered by the user.
        """


        # Check method fit has been called
        check_is_fitted(self)

        # check that input is a dataframe
        X = check_X(X)

        # Check input data contains same number of columns as df used to fit
        _check_X_matches_training_df(X, self.n_features_in_)

        # reorder df to match train set
        X = X[self.feature_names_in_]

        return X

    def transform(self, X: pl.DataFrame) -> pl.DataFrame:
        """Replace categories with the learned parameters.

        Parameters
        ----------
        X: polars or pandas dataframe of shape = [n_samples, n_features].
            The dataset to transform.

        Returns
        -------
        X_new: polars or pandas dataframe of shape = [n_samples, n_features].
            The dataframe containing the categories replaced by numbers.
        """

        X = self._check_transform_input_and_state(X)

        # check if dataset contains na
        if self.missing_values == "raise":
            _check_optional_contains_na(X, self.variables_)

        X = self._encode(X)

        return X

    def _encode(self, X: pl.DataFrame) -> pl.DataFrame:
        # replace categories by the learned parameters
        #print(self.encoder_dict_.values())
        for feature in self.encoder_dict_.keys():
            if X[feature].dtype != pl.Categorical:
                X = X.with_columns(pl.col(feature).replace_strict(self.encoder_dict_[feature], default = None
                                            ).alias(feature))
            else:
                X = X.with_columns(pl.col(feature).cast(pl.String).replace_strict(self.encoder_dict_[feature],
                                            default = None ).alias(feature))

        if self.unseen == "encode":
            X = X.fill_null(self._unseen)
        else:
            # check if nan values were introduced by the transformation
            self._check_nan_values_after_transformation(X)

        return X

    def _check_nan_values_after_transformation(self, X):

        # check if NaN values were introduced by the encoding
        if X[self.variables_].null_count().pipe(sum).item() > 0:

            # obtain the name(s) of the columns have null values
            null_cnt = X[self.variables_].null_count().to_dict(as_series = False)

            nan_columns = [x for x in null_cnt if null_cnt[x][0] > 0]

            if len(nan_columns) > 1:
                nan_columns_str = ", ".join(nan_columns)
            else:
                nan_columns_str = nan_columns[0]

            if self.unseen == "ignore":
                warnings.warn(
                    "During the encoding, NaN values were introduced in the feature(s) "
                    f"{nan_columns_str}."
                )
            elif self.unseen == "raise":
                raise ValueError(
                    "During the encoding, NaN values were introduced in the feature(s) "
                    f"{nan_columns_str}."
                )

    def inverse_transform(self, X: pl.DataFrame) -> pl.DataFrame:
        """Convert the encoded variable back to the original values.

        Parameters
        ----------
        X: polars or pandas dataframe of shape = [n_samples, n_features].
            The transformed dataframe.

        Returns
        -------
        X_tr: polars or pandas dataframe of shape = [n_samples, n_features].
            The un-transformed dataframe, with the categorical variables containing the
            original values.
        """

        X = self._check_transform_input_and_state(X)

        # replace encoded categories by the original values
        for feature in self.encoder_dict_.keys():
            inv_map = {v: k for k, v in self.encoder_dict_[feature].items()}
            #X[feature] = X[feature].map(inv_map)
            X = X.with_columns(pl.col(feature).cast(pl.String).replace(inv_map
                                                        ).alias(feature))
        return X

    def _more_tags(self):
        tags_dict = _return_tags()
        tags_dict["variables"] = "categorical"
        # the below test will fail because sklearn requires to check for inf, but
        # you can't check inf of categorical data, numpy returns and error.
        # so we need to leave without this test
        tags_dict["_xfail_checks"]["check_estimators_nan_inf"] = "transformer allows NA"
        return tags_dict
