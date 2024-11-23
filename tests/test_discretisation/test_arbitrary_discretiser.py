import numpy as np
import pandas as pd
import pytest
from numpy.random import default_rng
from scipy.stats import skewnorm
from sklearn.datasets import fetch_california_housing
import polars as pl
from polars.testing import assert_frame_equal
from fast_feature.discretisation import ArbitraryDiscretiser


def test_arbitrary_discretiser():
    california_dataset = fetch_california_housing()
    data = pl.DataFrame(
        california_dataset.data, schema=california_dataset.feature_names
    )
    user_dict = {"HouseAge": [0, 20, 40, 60]}

    data_t1 = data.clone()
    data_t2 = data.clone()

    # HouseAge is the median house age in the block group.
    data_t1 = data.with_columns(pl.col("HouseAge").cut(breaks =[0, 20, 40, 60]).alias('HouseAge').cast(pl.String))
    data_t2 = data.with_columns(pl.col("HouseAge").cut(breaks =[0, 20, 40, 60],  labels=['0', '1', '2', '3', '4']).alias('HouseAge').cast(pl.Int64))

    transformer = ArbitraryDiscretiser(
        binning_dict=user_dict, return_boundaries=False
    )
    X = transformer.fit_transform(data)

    # init params
    assert transformer.return_boundaries is False
    # fit params
    assert transformer.variables_ == ["HouseAge"]
    assert transformer.binner_dict_ == user_dict
    # transform params
    assert_frame_equal(X, data_t2)

    transformer = ArbitraryDiscretiser(
        binning_dict=user_dict, return_boundaries=True
    )
    X = transformer.fit_transform(data)
    assert_frame_equal(X, data_t1)


def test_error_if_input_df_contains_na_in_transform(df_vartypes, df_na):
    # test case 1: when dataset contains na, transform method
    age_dict = {"Age": [0, 10, 20, 30, np.inf]}

    with pytest.raises(ValueError):
        transformer = ArbitraryDiscretiser(binning_dict=age_dict)
        transformer.fit(df_vartypes)
        transformer.transform(df_na[["Name", "City", "Age", "Marks", "dob"]])


def test_error_if_not_permitted_value_is_errors():
    age_dict = {"Age": [0, 10, 20, 30, np.inf]}
    with pytest.raises(ValueError):
        ArbitraryDiscretiser(binning_dict=age_dict, errors="medialuna")


@pytest.mark.parametrize("binning_dict", ["HOLA", 1, False])
def test_error_if_binning_dict_not_dict_type(binning_dict):
    msg = (
        "binning_dict must be a dictionary with the interval limits per "
        f"variable. Got {binning_dict} instead."
    )
    with pytest.raises(ValueError) as record:
        ArbitraryDiscretiser(binning_dict=binning_dict)

    # check that error message matches
    assert str(record.value) == msg
