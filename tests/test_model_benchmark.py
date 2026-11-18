import numpy as np
import pandas as pd

from ml.benchmark_models import add_engineered_features, select_fold_features


def test_engineered_features_preserve_applicability_and_missing_status():
    source = pd.DataFrame(
        {
            "q314_pay_online_1": [1, 0],
            "q314_a_online_pay_ever_yn": [1, 0],
            "q314_b_online_pay_easy_diff": [np.nan, 2],
            "q466_fridge_yn": [1, 2],
            "q467_fridge_no": [1, np.nan],
            "q468_fridge_most_yrs": [3, np.nan],
            "q469_fridge_most_bee_star": [np.nan, 5],
            "q470_fridge_off_time": [2, np.nan],
            "survey_type_code": [1, 2],
            "q306_grid_rural_powercut_days_20": [4, np.nan],
            "q516_firewood_use_freq": [1, np.nan],
            "q517_firewood_collect_purchase": [2, np.nan],
            "q517_a_firewood_collect_who": [3, np.nan],
            "q517_b_firewood_collect_freq": [4, np.nan],
        }
    )

    enhanced = add_engineered_features(source)

    assert enhanced["online_payment_status"].tolist() == [
        "currently_pays_online",
        "response_code_2",
    ]
    assert enhanced["fridge_bee_status_experiment"].tolist() == [
        "not_reported",
        "not_applicable",
    ]
    assert enhanced["fridge_age_experiment"].tolist() == [3, 0]
    assert enhanced["rural_powercut_days_experiment"].tolist() == [4, 0]
    assert enhanced["firewood_use_frequency_experiment"].tolist() == [
        "code_1",
        "not_reported",
    ]


def test_fold_screening_uses_training_rows_and_drops_unstable_features():
    train = pd.DataFrame(
        {
            "keep": [0, 1] * 50,
            "mostly_missing": [np.nan] * 60 + [0, 1] * 20,
            "constant": [1] * 100,
            "near_constant": [1] * 99 + [0],
        }
    )
    validation = train.iloc[:10].copy()

    screened_train, screened_validation = select_fold_features(train, validation)

    assert screened_train.columns.tolist() == ["keep"]
    assert screened_validation.columns.tolist() == ["keep"]
