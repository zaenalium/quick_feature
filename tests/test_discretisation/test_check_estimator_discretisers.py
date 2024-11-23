import numpy as np
import pandas as pd
import polars as pl
from polars.testing import assert_frame_equal
import pytest
from sklearn.pipeline import Pipeline
from sklearn.utils.estimator_checks import check_estimator

from fast_feature.discretisation import (
    ArbitraryDiscretiser,
    DecisionTreeDiscretiser,
    EqualFrequencyDiscretiser,
    EqualWidthDiscretiser,
    GeometricWidthDiscretiser,
)
from tests.estimator_checks.estimator_checks import check_fast_feature_estimator

_estimators = [
    DecisionTreeDiscretiser(regression=False),
    EqualFrequencyDiscretiser(),
    EqualWidthDiscretiser(),
    ArbitraryDiscretiser(binning_dict={"x0": [0]}),
    GeometricWidthDiscretiser(),
]


@pytest.mark.parametrize("estimator", _estimators)
def test_check_estimator_from_sklearn(estimator):
    return check_estimator(estimator)


@pytest.mark.parametrize("estimator", _estimators)
def test_check_estimator_from_fast_feature(estimator):
    if estimator.__class__.__name__ == "ArbitraryDiscretiser":
        estimator.set_params(binning_dict={"var_1": [0]})
    return check_fast_feature_estimator(estimator)


@pytest.mark.parametrize("transformer", _estimators)
def test_transformers_within_pipeline(transformer):
    if transformer.__class__.__name__ == "ArbitraryDiscretiser":
        transformer.set_params(binning_dict={"feature_1": [0]})

    X = pl.DataFrame({"feature_1": [1, 2, 3, 4, 5], "feature_2": [6, 7, 8, 9, 10]})
    y = pl.Series([0, 1, 0, 1, 0])

    pipe = Pipeline([("trs", transformer)]).set_output(transform="polars")

    Xtt = transformer.fit_transform(X, y)
    Xtp = pipe.fit_transform(X, y)

    assert_frame_equal(Xtt, Xtp)
