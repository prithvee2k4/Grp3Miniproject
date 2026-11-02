import pandas as pd
import pytest

from etl import member2_structural_cleaning as m2

if not m2.IMPLEMENTED:
    pytest.skip("Member 2 not merged yet", allow_module_level=True)


def test_remove_duplicates():
    df = pd.DataFrame({"hhid": [1, 1, 2], "x": [5, 5, 6]})
    assert len(m2.remove_duplicates(df)) == 2


def test_drop_empty_columns():
    df = pd.DataFrame({"a": [1, 2, 3, 4], "b": [None, None, None, None]})
    assert list(m2.drop_empty_columns(df).columns) == ["a"]


def test_convert_dates_handles_both_formats():
    df = pd.DataFrame({"interview_date": ["12/18/2019", "12-10-2019"],
                       "interview_start_time": ["4:09 PM", "11:45 AM"]})
    out = m2.convert_dates(df)
    assert out["interview_date"].iloc[1] == pd.Timestamp("2019-12-10")
    assert list(out["interview_start_hour"]) == [16, 11]


def test_standardize_text():
    df = pd.DataFrame({"state": ["  UTTAR   PRADESH "], "state_abbr": ["up"]})
    out = m2.standardize_text(df)
    assert out["state"].iloc[0] == "Uttar Pradesh"
    assert out["state_abbr"].iloc[0] == "UP"
