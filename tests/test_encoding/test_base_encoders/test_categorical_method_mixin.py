import numpy as np
import pandas as pd
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from fast_feature.encoding.base_encoder import CategoricalMethodsMixin


class MockClassFit(CategoricalMethodsMixin):
    def __init__(self, missing_values="raise", ignore_format=False):
        self.missing_values = missing_values
        self.variables = None
        self.ignore_format = ignore_format


def test_underscore_check_na_method():
    input_df = pl.DataFrame(
        {
            "words": ["dog", "dig", "cat"],
            "animals": ["bird", "tiger", None],
        }
    )
    variables = ["words", "animals"]

    enc = MockClassFit(missing_values="raise")
    with pytest.raises(ValueError) as record:
        enc._check_na(input_df, variables)
    msg = (
        "Some of the variables in the dataset contain NaN. Check and "
        "remove those before using this transformer or set the parameter "
        "`missing_values='ignore'` when initialising this transformer."
    )
    assert str(record.value) == msg


def test_check_or_select_variables():
    input_df = pl.DataFrame(
        {
            "words": ["dog", "dig", "cat"],
            "animals": [1, 2, None],
        }
    )

    enc = MockClassFit(ignore_format=False)
    assert enc._check_or_select_variables(input_df) == ["words"]

    enc = MockClassFit(ignore_format=True)
    assert enc._check_or_select_variables(input_df) == ["words", "animals"]


def test_get_feature_names_in():
    input_df = pl.DataFrame(
        {
            "words": ["dog", "dig", "cat"],
            "animals": [1, 2, None],
        }
    )
    enc = MockClassFit()
    enc._get_feature_names_in(input_df)
    assert enc.feature_names_in_ == ["words", "animals"]
    assert enc.n_features_in_ == 2


class MockClass(CategoricalMethodsMixin):
    def __init__(self, unseen=None, missing_values="raise"):
        self.encoder_dict_ = {"words": {"dog": 1, "dig":2, "cat": 0}}
        self.n_features_in_ = 1
        self.feature_names_in_ = ["words"]
        self.variables_ = ["words"]
        self.missing_values = missing_values
        self.unseen = unseen
        self._unseen = -1

    def fit(self):
        return self


def test_transform_no_unseen():
    input_df = pl.DataFrame({"words": ["dog", "dig", "cat"]})
    output_df = pl.DataFrame({"words": [1,2, 0]}, strict = False)
    enc = MockClass()
    assert_frame_equal(enc.transform(input_df), output_df)


def test_transform_ignore_unseen():
    input_df = pl.DataFrame({"words": ["dog", "dig", "bird"]})
    output_df = pl.DataFrame({"words": [1,2, None]}, strict = False)
    enc = MockClass(unseen="ignore")
    assert_frame_equal(enc.transform(input_df), output_df)


def test_transform_encode_unseen():
    input_df = pl.DataFrame({"words": ["dog", "dig", "bird"]})
    output_df = pl.DataFrame({"words": [1,2, -1]}, strict = False)
    enc = MockClass(unseen="encode")
    assert_frame_equal(enc.transform(input_df), output_df)


def test_raises_error_when_nan_introduced():
    input_df = pl.DataFrame({"words": ["dog", "dig", "bird"]})
    output_df = pl.DataFrame({"words": [1,2, None]}, strict = False)
    enc = MockClass(unseen="raise")
    msg = "During the encoding, NaN values were introduced in the feature(s) words."

    with pytest.raises(ValueError) as record:
        enc._check_nan_values_after_transformation(output_df)
    assert str(record.value) == msg

    with pytest.raises(ValueError) as record:
        enc.transform(input_df)
    assert str(record.value) == msg


def test_raises_warning_when_nan_introduced():
    input_df = pl.DataFrame({"words": ["dog", "dig", "bird"]})
    output_df = pl.DataFrame({"words": [1,2, None]}, strict = False)
    enc = MockClass(unseen="ignore")
    msg = "During the encoding, NaN values were introduced in the feature(s) words."

    with pytest.warns(UserWarning) as record:
        enc.transform(input_df)
    assert record[0].message.args[0] == msg

    with pytest.warns(UserWarning) as record:
        enc._check_nan_values_after_transformation(output_df)
    assert record[0].message.args[0] == msg


def test_transform_raises_error_when_df_has_nan():
    input_df = pl.DataFrame({"words": ["dog", "dig", "cat", None]})
    enc = MockClass()
    with pytest.raises(ValueError) as record:
        enc.transform(input_df)
    msg = (
        "Some of the variables in the dataset contain NaN. Check and "
        "remove those before using this transformer or set the parameter "
        "`missing_values='ignore'` when initialising this transformer."
    )
    assert str(record.value) == msg


def test_transform_ignores_nan_in_df_to_transform():
    input_df = pl.DataFrame({"words": ["dog", "dig", "cat", None]}, strict = False)
    output_df = pl.DataFrame({"words": [1,2, 0, None]}, strict = False)
    enc = MockClass()
    enc.missing_values = "ignore"
    assert_frame_equal(enc.transform(input_df), output_df)


def test_inverse_transform_no_unseen_categories():
    input_df = pl.DataFrame({"words": ["dog", "dig", "cat"]})
    output_df = pl.DataFrame({"words": [1,2, 0]}, strict = False)

    # when no unseen categories
    enc = MockClass()
    assert_frame_equal(enc.inverse_transform(output_df), input_df)


def test_inverse_transform_when_ignore_unseen():
    input_df = pl.DataFrame({"words": ["dog", "dig", "bird"]})
    output_df = pl.DataFrame({"words": [1,2, None]}, strict = False)
    inverse_df = pl.DataFrame({"words": ["dog", "dig", None]})

    # when no unseen categories
    enc = MockClass(unseen="ignore")
    assert_frame_equal(enc.transform(input_df), output_df)
    assert_frame_equal(enc.inverse_transform(output_df), inverse_df)
