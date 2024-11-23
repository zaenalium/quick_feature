import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

from quick_feature.discretisation import EqualWidthDiscretiser


def test_automatically_find_variables_and_return_as_numeric(df_normal_dist):
    # test case 1: automatically select variables, return_object=False
    transformer = EqualWidthDiscretiser(bins=10, variables=None)
    X = transformer.fit_transform(df_normal_dist)

    # fit parameters
    _, bins = pd.cut(x=df_normal_dist["var"], bins=10, retbins=True, duplicates="drop")
    bins = bins[1:-1]
    #bins[0] = float("-inf")
    #bins[len(bins) - 1] = float("inf")

    # transform output
    X_t = [x for x in range(0, 10)]
    val_counts = [18, 17, 16, 13, 11, 7, 7, 5, 5, 1]

    # init params
    assert transformer.bins == 10
    assert transformer.variables is None
    # fit params
    assert transformer.variables_ == ["var"]
    assert transformer.n_features_in_ == 1
    # transform params
    assert (transformer.binner_dict_["var"] == bins).all()
    assert all(x for x in X["var"].unique() if x not in X_t)
    # in equal width discretisation, intervals get different number of values
    assert all(x for x in X["var"].value_counts()['count'] if x not in val_counts)


def test_error_when_bins_not_number():
    with pytest.raises(ValueError):
        EqualWidthDiscretiser(bins="other")


def test_error_if_input_df_contains_na_in_fit(df_na):
    # test case 3: when dataset contains na, fit method
    with pytest.raises(ValueError):
        transformer = EqualWidthDiscretiser()
        transformer.fit(df_na)


def test_error_if_input_df_contains_na_in_transform(df_vartypes, df_na):
    # test case 4: when dataset contains na, transform method
    with pytest.raises(ValueError):
        transformer = EqualWidthDiscretiser()
        transformer.fit(df_vartypes)
        transformer.transform(df_na[["Name", "City", "Age", "Marks", "dob"]])


def test_non_fitted_error(df_vartypes):
    with pytest.raises(NotFittedError):
        transformer = EqualWidthDiscretiser()
        transformer.transform(df_vartypes)
