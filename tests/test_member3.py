import numpy as np
import pandas as pd
import pytest

from etl import member3_missing_invalid as m3

if not m3.IMPLEMENTED:
    pytest.skip("Member 3 not merged yet", allow_module_level=True)


def test_dont_know_codes_become_missing():
    df = pd.DataFrame({"q304_x": [1, 2, 99, 3, 1], "income_category": [1, 88, 2, 3, 1]})
    out = m3.replace_dont_know_codes(df)
    assert out["q304_x"].isna().sum() == 1
    assert out["income_category"].isna().sum() == 1


def test_cap_outliers():
    df = pd.DataFrame({"land_area_sqft": list(range(1, 200)) + [10_000_000]})
    out = m3.cap_outliers(df)
    assert out["land_area_sqft"].max() < 1000


def test_fill_missing_uses_state_median():
    df = pd.DataFrame({"state": ["A", "A", "A", "B"],
                       "land_area_sqft": [10.0, 30.0, np.nan, 100.0],
                       "monthly_expenditure": [1.0, 2.0, 3.0, 4.0],
                       "avg_monthly_elec_bill": [1.0, 2.0, 3.0, 4.0]})
    out = m3.fill_missing_values(df)
    assert out["land_area_sqft"].iloc[2] == 20.0
    assert out["land_area_sqft_was_missing"].iloc[2]
