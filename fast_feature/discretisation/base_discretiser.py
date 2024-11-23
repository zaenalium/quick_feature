# Authors: Morgan Sell <morganpsell@gmail.com>
# License: BSD 3 clause

import pandas as pd
import polars as pl

from fast_feature._base_transformers.base_numerical import BaseNumericalTransformer


class BaseDiscretiser(BaseNumericalTransformer):
    """
    Shared set-up checks and methods across numerical discretisers.

    Important: inherits fit() functionality and tags from BaseNumericalTransformer.
    """

    def __init__(
        self,
        return_boundaries: bool = False,
    ) -> None:

        if not isinstance(return_boundaries, bool):
            raise ValueError(
                "return_boundaries must be True or False. "
                f"Got {return_boundaries} instead."
            )

        self.return_boundaries = return_boundaries

    def transform(self, X: pl.DataFrame) -> pl.DataFrame:
        """Sort the variable values into the intervals.

        Parameters
        ----------
        X: polars or pandas dataframe of shape = [n_samples, n_features]
            The data to transform.

        Returns
        -------
        X_new: polars or pandas dataframe of shape = [n_samples, n_features]
            The transformed data with the discrete variables.
        """

        # check input dataframe and if class was fitted
        X = self._check_transform_input_and_state(X)

        # transform variables
        if self.return_boundaries is True:
            for feature in self.variables_:
                X = X.with_columns(
                   pl.col(feature).cut(self.binner_dict_[feature]).alias(feature)
                )
            X[self.variables_] = X[self.variables_].cast(pl.String)
        else:
            for feature in self.variables_:
                labs = [str(x) for x in range(len(self.binner_dict_[feature]) + 1)]
                X = X.with_columns(
                   pl.col(feature).cut(self.binner_dict_[feature], labels = labs).alias(feature)
                )

            X[self.variables_] = X[self.variables_].cast(pl.Int64)

        return X
