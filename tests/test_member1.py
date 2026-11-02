import pandas as pd
import pytest

from etl import extract_step

if not extract_step.IMPLEMENTED:
    pytest.skip("Member 1 not merged yet", allow_module_level=True)


def test_validate_raw_rejects_empty():
    with pytest.raises(ValueError):
        extract_step.validate_raw(pd.DataFrame())


def test_validate_raw_rejects_missing_columns():
    with pytest.raises(ValueError):
        extract_step.validate_raw(pd.DataFrame({"hhid": [1]}))


def test_extract_reads_real_file():
    df = extract_step.run_extract(save_copy=False)
    assert df.shape == (14851, 517)
