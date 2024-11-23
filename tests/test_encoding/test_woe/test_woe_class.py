import numpy as np
import pandas as pd
import pytest
import polars as pl
from polars.testing import assert_frame_equal

from fast_feature.encoding.woe import WoE


def test_woe_calculation(df_enc):
    pos_exp = pl.DataFrame({
        'var_A' :  ['A','B', 'C'],
        'pos' : [0.333333, 0.333333,  0.333333],
        'neg' : [ 0.285714,  0.571429,  0.142857]      
    })

    pos_exp = pos_exp.with_columns(woe = np.log(pl.col('pos') / pl.col('neg')))
    woe_class = WoE()
    woe = woe_class._calculate_woe(df_enc, df_enc["target"], "var_A")
    assert_frame_equal(woe.sort(by= 'var_A'), pos_exp.sort(by= 'var_A'))



def test_woe_error():
    df = {
        "var_A": ["B"] * 9 + ["A"] * 6 + ["C"] * 3 + ["D"] * 2,
        "var_B": ["A"] * 10 + ["B"] * 6 + ["C"] * 4,
        "target": [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0],
    }
    df = pl.DataFrame(df)
    woe_class = WoE()

    with pytest.raises(ValueError):
        woe_class._calculate_woe(df, df["target"], "var_A")


@pytest.mark.parametrize("fill_value", [1, 10, 0.1])
def test_fill_value(fill_value):
    df = {
        "var_A": ["A"] * 9 + ["B"] * 6 + ["C"] * 3 + ["D"] * 2,
        "var_B": ["A"] * 10 + ["B"] * 6 + ["C"] * 4,
        "target": [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0],
    }
    df = pl.DataFrame(df)

    pos_exp = pl.DataFrame({
        'var_A' :  ['A','B', 'C', "D"],
        'pos' : [0.2857142857142857, 0.2857142857142857,  0.42857142857142855, fill_value],
        'neg' : [  0.5384615384615384, 0.3076923076923077, fill_value, 0.15384615384615385,]
    })

    pos_exp = pos_exp.with_columns(woe = np.log(pl.col('pos') / pl.col('neg')))

    woe_class = WoE()
    woe = woe_class._calculate_woe(
        df, df["target"], "var_A", fill_value=fill_value
    )
    assert_frame_equal(woe.sort(by= 'var_A'), pos_exp.sort(by= 'var_A'))
