import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

from fast_feature.discretisation import GeometricWidthDiscretiser
import polars as pl

# test init params

@pytest.mark.parametrize("param", [0.1, (True, False)])
def test_raises_error_when_return_boundaries_not_bool(param):
    with pytest.raises(ValueError):
        GeometricWidthDiscretiser(return_boundaries=param)


@pytest.mark.parametrize("param", [0.1, (True, False)])
def test_raises_error_when_bins_not_int(param):
    with pytest.raises(ValueError):
        GeometricWidthDiscretiser(bins=param)


@pytest.mark.parametrize("params", [(False, 1), (True, 10)])
def test_correct_param_assignment_at_init(params):
    param1, param2 = params
    t = GeometricWidthDiscretiser(
        return_boundaries=param1, bins=param2
    )
    assert t.return_boundaries is param1
    assert t.bins == param2


def test_fit_and_transform_methods(df_normal_dist):
    transformer = GeometricWidthDiscretiser(
        bins=10, variables=None
    )
    X = transformer.fit_transform(df_normal_dist)

    # manual calculation
    min_, max_ = df_normal_dist["var"].min(), df_normal_dist["var"].max()
    increment = np.power(max_ - min_, 1.0 / 10)
    bins = np.r_[min_ + np.power(increment, np.arange(1, 10))]
    bins = np.sort(bins)

    # fit params
    assert (transformer.binner_dict_["var"] == bins).all()

    # transform params
    labs = [str(x) for x in range(len(bins) + 1)]
    assert (
        X["var"] == pl.from_pandas(df_normal_dist["var"]).cut(bins,labels = labs).cast(pl.Int64())
    ).all()



def test_error_if_input_df_contains_na_in_fit(df_na):
    # test case 3: when dataset contains na, fit method
    transformer = GeometricWidthDiscretiser()
    with pytest.raises(ValueError):
        transformer.fit(df_na)


def test_error_if_input_df_contains_na_in_transform(df_vartypes, df_na):
    # test case 4: when dataset contains na, transform method
    transformer = GeometricWidthDiscretiser()
    transformer.fit(df_vartypes)
    with pytest.raises(ValueError):
        transformer.transform(df_na[["Name", "City", "Age", "Marks", "dob"]])


def test_non_fitted_error(df_vartypes):
    transformer = GeometricWidthDiscretiser()
    with pytest.raises(NotFittedError):
        transformer.transform(df_vartypes)
