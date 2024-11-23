

from typing import List, Optional, Union

import pandas as pd
import polars as pl
from quick_feature._docstrings.fit_attributes import (
    _feature_names_in_docstring,
    _n_features_in_docstring,
    _variables_attribute_docstring,
)
from quick_feature._docstrings.init_parameters.all_trasnformers import (
    _missing_values_docstring,
    _variables_categorical_docstring,
)
from quick_feature._docstrings.init_parameters.encoders import (
    _ignore_format_docstring,
    _unseen_docstring,
)
from quick_feature._docstrings.methods import (
    _fit_transform_docstring,
    _inverse_transform_docstring,
    _transform_encoders_docstring,
)
from quick_feature._docstrings.substitute import Substitution
from quick_feature.dataframe_checks import check_X, check_X_y
from quick_feature.encoding._helper_functions import check_parameter_unseen
from quick_feature.encoding.base_encoder import (
    CategoricalInitMixinNA,
    CategoricalMethodsMixin,
)

_unseen_docstring = (
    _unseen_docstring + """ If `'encode'`, unseen categories will be encoded as -1."""
)


@Substitution(
    missing_values=_missing_values_docstring,
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
class OrdinalEncoder(CategoricalInitMixinNA, CategoricalMethodsMixin):
    """
    The OrdinalEncoder() replaces categories by ordinal numbers
    (0, 1, 2, 3, etc). The numbers can be ordered based on the mean of the target
    per category, or assigned arbitrarily.

    The encoder will encode only categorical variables by default (type 'object' or
    'categorical'). You can pass a list of variables to encode. Alternatively, the
    encoder will find and encode all categorical variables (type 'object' or
    'categorical').

    With `ignore_format=True` you have the option to encode numerical variables as well.
    The procedure is identical, you can either enter the list of variables to encode, or
    the transformer will automatically select all variables.

    The encoder first maps the categories to the numbers for each variable (fit). The
    encoder then transforms the categories to the mapped numbers (transform).

    Parameters
    ----------
    encoding_method: str, default='ordered'
        Desired method of encoding.

        **'ordered'**: the categories are numbered in ascending order according to
        the target mean value per category.

        **'arbitrary'**: categories are numbered arbitrarily.

    {variables}

    {missing_values}

    {ignore_format}

    {unseen}

    Attributes
    ----------
    encoder_dict_:
        Dictionary with the ordinal number per category, per variable.

    {variables_}

    {feature_names_in_}

    {n_features_in_}

    Methods
    -------
    fit:
        Find the integer to replace each category in each variable.

    {fit_transform}

    {inverse_transform}

    {transform}

    Notes
    -----
    NAN are introduced when encoding categories that were not present in the training
    dataset. If this happens, try grouping infrequent categories using the
    RareLabelEncoder().

    There is a similar implementation in the the open-source package
    `Category encoders <https://contrib.scikit-learn.org/category_encoders/>`_

    See Also
    --------
    quick_feature.encoding.RareLabelEncoder
    category_encoders.ordinal.OrdinalEncoder

    References
    ----------
    Encoding into integers ordered following target mean was discussed in the following
    talk at PyData London 2017:

    .. [1] Galli S. "Machine Learning in Financial Risk Assessment".
        https://www.youtube.com/watch?v=KHGGlozsRtA

    Examples
    --------

    >>> import pandas as pd
    >>> import polars as pl
    >>> from quick_feature.encoding import OrdinalEncoder
    >>> X = pd.DataFrame(dict(x1 = [1,2,3,4], x2 = ["c", "a", "b", "c"])) # X = pl.DataFrame(dict(x1 = [1,2,3,4], x2 = ["c", "a", "b", "c"]))
    >>> y = pl.Series([0,1,1,0])
    >>> od = OrdinalEncoder(encoding_method='arbitrary')
    >>> od.fit(X)
    >>> od.transform(X)
       x1  x2
    0   1   0
    1   2   1
    2   3   2
    3   4   0

    You can also consider the order of the target variable:

    >>> y = pl.Series([1,0,1,1])
    >>> od = OrdinalEncoder(encoding_method='ordered')
    >>> od.fit(X, y)
    >>> od.transform(X)
       x1  x2
    0   1   2
    1   2   0
    2   3   1
    3   4   2
    """

    def __init__(
        self,
        encoding_method: str = "ordered",
        variables: Union[None, int, str, List[Union[str, int]]] = None,
        missing_values: str = "raise",
        ignore_format: bool = False,
        unseen: str = "ignore",
    ) -> None:

        if encoding_method not in ["ordered", "arbitrary"]:
            raise ValueError(
                "encoding_method takes only values 'ordered' and 'arbitrary'"
            )

        check_parameter_unseen(unseen, ["ignore", "raise", "encode"])
        super().__init__(variables, missing_values, ignore_format)
        self.encoding_method = encoding_method
        self.unseen = unseen

    def fit(self, X: pl.DataFrame, y: Optional[pl.Series] = None):
        """Learn the numbers to be used to replace the categories in each
        variable.

        Parameters
        ----------
        X: polars or pandas dataframe of shape = [n_samples, n_features]
            The training input samples. Can be the entire dataframe, not just the
            variables to be encoded.

        y: polars or pandas series, default=None
            The Target. Can be None if `encoding_method='arbitrary'`.
            Otherwise, y needs to be passed when fitting the transformer.
        """

        if self.encoding_method == "ordered":
            X, y = check_X_y(X, y)
        else:
            X = check_X(X)

        variables_ = self._check_or_select_variables(X)
        self._check_na(X, variables_)

        self.encoder_dict_ = {}
        
        

        for var in variables_:
            if self.encoding_method == "ordered":
                X = X.with_columns(y = y)
                t = X.group_by(var).agg(pl.col("y").mean())
                X = X.drop('y')
                t = t.sort('y',descending=False)
                t = t[var]
            elif self.encoding_method == "arbitrary":
                if self.missing_values == "ignore":
                    t = X[var].drop_nulls().drop_nans().unique(maintain_order = True)
                else:
                    t = X[var].unique(maintain_order = True)
            else:
                raise ValueError(
                    "Unrecognized value for encoding_method. It should be 'arbitrary' "
                    f"or 'frequency'. Got {self.encoding_method} instead."
                )

            self.encoder_dict_[var] = {k: i for i, k in enumerate(t, 0)}

        if self.unseen == "encode":
            self._unseen = -1

        # assign underscore parameters at the end in case code above fails
        self.variables_ = variables_
        self._get_feature_names_in(X)
        return self
