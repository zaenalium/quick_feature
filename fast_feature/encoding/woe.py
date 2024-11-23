

from typing import List, Union

import numpy as np
import pandas as pd
import polars as pl
from fast_feature._docstrings.fit_attributes import (
    _feature_names_in_docstring,
    _n_features_in_docstring,
    _variables_attribute_docstring,
)
from fast_feature._docstrings.init_parameters.all_trasnformers import (
    _variables_categorical_docstring,
)
from fast_feature._docstrings.init_parameters.encoders import (
    _ignore_format_docstring,
    _unseen_docstring,
)
from fast_feature._docstrings.methods import (
    _fit_transform_docstring,
    _inverse_transform_docstring,
    _transform_encoders_docstring,
)
from fast_feature._docstrings.substitute import Substitution
from fast_feature.dataframe_checks import _check_contains_na, check_X_y
from fast_feature.encoding._helper_functions import check_parameter_unseen
from fast_feature.encoding.base_encoder import (
    CategoricalInitMixin,
    CategoricalMethodsMixin,
)
from fast_feature.tags import _return_tags


class WoE:
    def _check_fit_input(self, X: pl.DataFrame, y: pl.Series):
        """
        Check that X is dataframe, and y a binary series with values 0 and 1.
        """
        X, y = check_X_y(X, y)

        # check that y is binary
        if y.n_unique() != 2:
            raise ValueError(
                "This encoder is designed for binary classification. The target "
                "used has more than 2 unique values."
            )

        # if target does not have values 0 and 1, we need to remap, to be able to
        # compute the averages.
        if y.min() != 0 or y.max() != 1:
            y = pl.Series(np.where(y == y.min(), 0, 1))
        return X, y

    def _calculate_woe(
        self,
        X: pl.DataFrame,
        y: pl.Series,
        variable: Union[str, int],
        fill_value: Union[float, None] = None,
    ):
        X, y = self._check_fit_input(X, y)
        total_pos = y.sum()
        inverse_y = y.ne(1)
        total_neg = inverse_y.sum()

        # pos = y.groupby(X[variable], observed=False).sum() / total_pos
        # neg = inverse_y.groupby(X[variable], observed=False).sum() / total_neg
        pos = X.with_columns(pos= y).group_by(variable).agg(pl.col('pos').sum()/total_pos)
        neg = X.with_columns(neg= inverse_y).group_by(variable).agg(pl.col('neg').sum()/total_neg)
        
        # pos = pos.to_pandas().set_index(variable)['pos']
        # neg = neg.to_pandas().set_index(variable)['pos']

        if  (pos['pos']==0).sum() + (neg['neg']==0).sum() > 0:
            if fill_value is None:
                raise ValueError(
                    "The proportion of one of the classes for a category in "
                    "variable {} is zero, and log of zero is not defined".format(
                        variable
                    )
                )
            else:
                pos = pos.with_columns(pl.when(pl.col("pos")  == 0
                                               ).then(fill_value).otherwise((pl.col("pos")))\
                                               .alias("pos"))
                neg = neg.with_columns(pl.when(pl.col("neg")  == 0
                                               ).then(fill_value).otherwise((pl.col("neg")))\
                                               .alias("neg"))
        woe = pos.join(neg, on = variable)
        woe = woe.with_columns(woe = np.log(pl.col('pos') / pl.col('neg')))
        return woe

@Substitution(
    ignore_format=_ignore_format_docstring,
    variables=_variables_categorical_docstring,
    unseen=_unseen_docstring,
    variables_=_variables_attribute_docstring,
    feature_names_in_=_feature_names_in_docstring,
    n_features_in_=_n_features_in_docstring,
    fit_transform=_fit_transform_docstring,
    transform=_transform_encoders_docstring,
    inverse_transform=_inverse_transform_docstring,
)
class WoEEncoder(CategoricalInitMixin, CategoricalMethodsMixin, WoE):
    """
    The WoEEncoder() replaces categories by the weight of evidence
    (WoE). The WoE was used primarily in the financial sector to create credit risk
    scorecards.

    The encoder will encode only categorical variables by default
    (type 'object' or 'categorical'). You can pass a list of variables to encode.
    Alternatively, the encoder will find and encode all categorical variables
    (type 'object' or 'categorical').

    With `ignore_format=True` you have the option to encode numerical variables as well.
    The procedure is identical, you can either enter the list of variables to encode, or
    the transformer will automatically select all variables.

    The encoder first maps the categories to the weight of evidence for each variable
    (fit). The encoder then transforms the categories into the mapped numbers
    (transform).

    This categorical encoding is exclusive for binary classification.

    **Note**

    The log(0) is not defined and the division by 0 is not defined. Thus, if any of the
    terms in the WoE equation are 0 for a given category, the encoder will return an
    error. If this happens, try grouping less frequent categories. Alternatively,
    you can now add a fill_value (see parameter below).

    More details in the :ref:`User Guide <woe_encoder>`.

    Parameters
    ----------
    {variables}

    {ignore_format}

    {unseen}

    fill_value: int, float, default=None
        When the numerator or denominator of the WoE calculation are zero, the WoE
        calculation is not possible. If `fill_value` is None (recommended), an error
        will be raised in those cases. Alternatively, fill_value will be used in place
        of denominators or numerators that equal zero.

    Attributes
    ----------
    encoder_dict_:
        Dictionary with the WoE per variable.

    {variables_}

    {feature_names_in_}

    {n_features_in_}

    Methods
    -------
    fit:
        Learn the WoE per category, per variable.

    {transform}

    {fit_transform}

    {inverse_transform}

    Notes
    -----
    For details on the calculation of the weight of evidence visit:
    https://www.listendata.com/2015/03/weight-of-evidence-woe-and-information.html

    NAN are introduced when encoding categories that were not present in the training
    dataset. If this happens, try grouping infrequent categories using the
    RareLabelEncoder().

    There is a similar implementation in the the open-source package
    `Category encoders <https://contrib.scikit-learn.org/category_encoders/>`_

    See Also
    --------
    fast_feature.encoding.RareLabelEncoder
    fast_feature.discretisation
    category_encoders.woe.WOEEncoder

    Examples
    --------

    >>> import pandas as pd
    >>> from fast_feature.encoding import WoEEncoder
    >>> X = pl.DataFrame(dict(x1 = [1,2,3,4,5], x2 = ["b", "b", "b", "a", "a"]))
    >>> y = pl.Series([0,1,1,1,0])
    >>> woe = WoEEncoder()
    >>> woe.fit(X, y)
    >>> woe.transform(X)
       x1        x2
    0   1  0.287682
    1   2  0.287682
    2   3  0.287682
    3   4 -0.405465
    4   5 -0.405465
    """

    def __init__(
        self,
        variables: Union[None, int, str, List[Union[str, int]]] = None,
        ignore_format: bool = False,
        unseen: str = "ignore",
        fill_value: Union[int, float, None] = None,
    ) -> None:

        super().__init__(variables, ignore_format)
        check_parameter_unseen(unseen, ["ignore", "raise"])
        if fill_value is not None and not isinstance(fill_value, (int, float)):
            raise ValueError(
                f"fill_value takes None, integer or float. Got {fill_value} instead."
            )
        self.unseen = unseen
        self.fill_value = fill_value

    def fit(self, X: pl.DataFrame, y: pl.Series):
        """
        Learn the WoE.

        Parameters
        ----------
        X: polars or pandas dataframe of shape = [n_samples, n_features]
            The training input samples.
            Can be the entire dataframe, not just the categorical variables.

        y: pandas series.
            Target, must be binary.
        """
        X, y = self._check_fit_input(X, y)
        variables_ = self._check_or_select_variables(X)
        _check_contains_na(X, variables_)

        encoder_dict_ = {}
        vars_that_fail = []

        for var in variables_:
            try:
                woe = self._calculate_woe(X, y, var, self.fill_value)
                var_woe = woe.to_pandas()[[var,'woe']]
                var_woe = var_woe.set_index(var)['woe']
                encoder_dict_[var] = var_woe.to_dict()
            except ValueError:
                vars_that_fail.append(var)

        if len(vars_that_fail) > 0:
            vars_that_fail_str = (
                ", ".join(vars_that_fail)
                if len(vars_that_fail) > 1
                else vars_that_fail[0]
            )

            raise ValueError(
                "During the WoE calculation, some of the categories in the "
                "following features contained 0 in the denominator or numerator, "
                f"and hence the WoE can't be calculated: {vars_that_fail_str}."
            )

        self.encoder_dict_ = encoder_dict_
        self.variables_ = variables_
        self._get_feature_names_in(X)
        return self

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
        _check_contains_na(X, self.variables_)
        X = self._encode(X)
        return X

    def _more_tags(self):
        tags_dict = _return_tags()
        tags_dict["variables"] = "categorical"
        tags_dict["requires_y"] = True
        # in the current format, the tests are performed using continuous np.arrays
        # this means that when we encode some of the values, the denominator is 0
        # and this the transformer raises an error, and the test fails.
        # For this reason, most sklearn transformers will fail. And it has nothing to
        # do with the class not being compatible, it is just that the inputs passed
        # are not suitable
        tags_dict["_skip_test"] = True
        return tags_dict
