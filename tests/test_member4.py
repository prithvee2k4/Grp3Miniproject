import pandas as pd
import pytest

from etl import member4_features as m4

if not m4.IMPLEMENTED:
    pytest.skip("Member 4 not merged yet", allow_module_level=True)


def _df():
    return pd.DataFrame({
        "hhid": [1, 2, 3, 4], "state": ["B", "A", "A", "B"], "district": ["x", "y", "z", "w"],
        "household_size": [2, 4, 8, 5], "avg_monthly_elec_bill": [100.0, 400.0, 900.0, 200.0],
        "monthly_expenditure": [1000.0, 4000.0, 9000.0, 2000.0], "bedrooms": [1, 2, 2, 1],
        "members_under_18": [0, 2, 4, 1], "respondent_age": [20, 40, 50, 65],
        "owns_house": [1, 0, 1, 1]})


def test_calculated_columns():
    out = m4.add_calculated_columns(_df())
    assert out["elec_bill_per_member"].iloc[0] == 50.0
    assert out["bill_share_of_expenditure_pct"].iloc[0] == 10.0


def test_categories():
    out = m4.add_categories(_df())
    assert list(out["age_group"].astype(str)) == ["18-29", "30-44", "45-59", "60+"]
    assert out["household_size_group"].astype(str).iloc[2] == "Large (7+)"


def test_ranking_and_sort():
    out = m4.sort_records(m4.add_rankings(_df()))
    assert out["state"].iloc[0] == "A"
    assert out.loc[out["state"] == "A", "state_bill_rank"].iloc[0] == 1
