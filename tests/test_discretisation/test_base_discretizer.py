import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import fetch_california_housing
import polars as pl

from fast_feature.discretisation.base_discretiser import BaseDiscretiser


# test init params

@pytest.mark.parametrize("param", [0.1, "hola", (True, False), {"a": True}, 2])
def test_raises_error_when_return_boundaries_not_bool(param):
    with pytest.raises(ValueError):
        BaseDiscretiser(return_boundaries=param)


@pytest.mark.parametrize("params", [(False, 1), (True, 10)])
def test_correct_param_assignment_at_init(params):
    param1, param2 = params
    t = BaseDiscretiser(
        return_boundaries=param1
    )
    assert t.return_boundaries is param1
