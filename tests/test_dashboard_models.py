import numpy as np
import pandas as pd
import pytest

from ml.dashboard_models import (
    BILL_TARGET,
    CONSUMPTION_TARGET,
    FEATURES,
    observed_bill_mask,
    prepare_training_data,
)


def sample_data() -> pd.DataFrame:
    data = {column: [1, 2, 3] for column in FEATURES}
    data.update(
        {
            "state": ["A", "A", "B"],
            BILL_TARGET: [100.0, 200.0, 300.0],
            CONSUMPTION_TARGET: [10.0, np.nan, 30.0],
            "avg_monthly_elec_bill_was_missing": [False, True, False],
            "q610_c_billed_amount": [100.0, 200.0, 300.0],
            "q315_a_bill_mar_jun": [100.0, 200.0, 300.0],
        }
    )
    return pd.DataFrame(data)


def test_bill_training_data_excludes_missing_target_rows_and_bill_leaks():
    features, target, groups = prepare_training_data(sample_data(), BILL_TARGET)

    assert target.tolist() == [100.0, 300.0]
    assert groups.tolist() == ["A", "B"]
    assert set(features.columns) == set(FEATURES)
    assert "q610_c_billed_amount" not in features
    assert "q315_a_bill_mar_jun" not in features


def test_consumption_training_data_uses_only_observed_billed_units():
    features, target, groups = prepare_training_data(
        sample_data(), CONSUMPTION_TARGET
    )

    assert target.tolist() == [10.0, 30.0]
    assert groups.tolist() == ["A", "B"]
    assert set(features.columns) == set(FEATURES)


def test_training_data_rejects_unknown_target():
    with pytest.raises(ValueError, match="Unsupported dashboard target"):
        prepare_training_data(sample_data(), "unknown_target")


def test_bill_comparisons_can_identify_originally_observed_rows():
    observed = observed_bill_mask(sample_data())

    assert observed.tolist() == [True, False, True]
