"""Compare HGB, CatBoost, and XGBoost on leakage-aware cross-validation folds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "output" / "analysis_ready_data.csv"
OUTPUT_DIR = ROOT / "output" / "ml_preprocessed"
TARGET = "avg_monthly_elec_bill"
TARGET_MISSING_FLAG = "avg_monthly_elec_bill_was_missing"
RANDOM_STATE = 42

LEAKAGE_COLUMNS = {
    "q315_a_bill_mar_jun",
    "q315_b_bill_jul_oct",
    "q315_c_bill_nov_feb",
    "elec_bill_per_member",
    "bill_share_of_expenditure_pct",
    "bill_group",
    "state_bill_rank",
    "household_bill_rank",
    TARGET_MISSING_FLAG,
    "q610_c_billed_amount",
    "monthly_expenditure",
    "monthly_expenditure_was_missing",
    "expenditure_per_member",
    "expenditure_group",
    "q608_b_ifprep_elec_bill_change",
    "q608_c_ifprep_wrong_bill",
}

SENSITIVE_COLUMNS = {
    "q210_religion",
    "caste_code",
    "respondent_gender_code",
    "q204_priminc_earner_gender",
}

METADATA_COLUMNS = {
    "enumerator_id",
    "state_abbr",
    "district",
    "district_code",
    "village_ward",
    "village_ward_census_code",
    "interview_date",
    "interview_start_time",
    "age_group",
    "household_size_group",
    "hhid",
    "state_code",
    "sw_state",
    "sw_dist",
}

CATEGORICAL_COLUMNS = {
    "state",
    "q203_priminc_earner_who",
    "q210_religion",
    "caste_code",
    "q212_ration_card",
    "q216_house_pucca_kachha",
    "q233_b_drink_main_watersource",
    "q304_grid_patternpowercut",
    "q312_grid_elec_meter_yn",
    "q313_grid_billfreq",
    "q414_a_bee_ceiling_fans_heard_yn",
    "q461_tv_most_type",
    "q505_lpg_large_small_both",
    "q529_prim_cook_fuel",
    "q601_appl_purch_dec_who",
    "q602_appl_pur_imp_para1",
    "q602_appl_pur_imp_para2",
    "q602_appl_pur_imp_para3",
    "q602_appl_pur_imp_para4",
    "survey_type_code",
    "q217_house_type",
    "land_area_sqfeet_reportd_estimtd",
    "q414_ceiling_fan_mostused_brand",
    "q208_priminc_earner_edu",
    "q209_edu_highest",
    "income_category",
    "q326_satis_electricity",
    "q530_satis_prim_cook_arrange",
    "q531_satis_prim_cook_fuel",
    "asset_decile_1",
    "asset_decile_2",
    "land_area_sqft_was_missing",
}

ENGINEERED_CATEGORICAL = {
    "online_payment_status",
    "fridge_bee_status_experiment",
    "fridge_off_time_status_experiment",
    "firewood_use_frequency_experiment",
    "firewood_collect_purchase_experiment",
    "firewood_collect_who_experiment",
    "firewood_collect_frequency_experiment",
}
ENGINEERED_NUMERIC = {
    "fridge_count_experiment",
    "fridge_age_experiment",
    "rural_powercut_days_experiment",
}
ENGINEERED_COLUMNS = ENGINEERED_CATEGORICAL | ENGINEERED_NUMERIC


def code_category(value: Any) -> str:
    if pd.isna(value):
        return "not_reported"
    return f"code_{int(value)}"


def conditional_code_category(values: pd.Series, applicable: pd.Series) -> pd.Series:
    result = []
    for value, is_applicable in zip(values, applicable):
        if not is_applicable:
            result.append("not_applicable")
        elif pd.isna(value):
            result.append("not_reported")
        else:
            result.append(f"code_{int(value)}")
    return pd.Series(result, index=values.index, dtype="object")


def add_engineered_features(features: pd.DataFrame) -> pd.DataFrame:
    """Recreate the notebook's ten rule-based features without using the target."""
    enhanced = features.copy()

    def online_status(row: pd.Series) -> str:
        current = row["q314_pay_online_1"]
        previous = row["q314_a_online_pay_ever_yn"]
        response = row["q314_b_online_pay_easy_diff"]
        if current == 1:
            return (
                "inconsistent_response"
                if pd.notna(response)
                else "currently_pays_online"
            )
        if current == 0:
            if previous == 1:
                return "previously_paid_online"
            if previous == 0:
                return (
                    "eligible_no_response"
                    if pd.isna(response)
                    else f"response_code_{int(response)}"
                )
        return "unknown_or_missing_status"

    enhanced["online_payment_status"] = enhanced.apply(online_status, axis=1)

    is_fridge_owner = enhanced["q466_fridge_yn"].eq(1)
    enhanced["fridge_count_experiment"] = enhanced["q467_fridge_no"].fillna(0)
    enhanced["fridge_age_experiment"] = enhanced["q468_fridge_most_yrs"].where(
        is_fridge_owner, 0
    )
    enhanced["fridge_bee_status_experiment"] = conditional_code_category(
        enhanced["q469_fridge_most_bee_star"], is_fridge_owner
    )
    enhanced["fridge_off_time_status_experiment"] = conditional_code_category(
        enhanced["q470_fridge_off_time"], is_fridge_owner
    )

    is_rural = enhanced["survey_type_code"].eq(1)
    enhanced["rural_powercut_days_experiment"] = (
        enhanced["q306_grid_rural_powercut_days_20"]
    )
    enhanced.loc[~is_rural, "rural_powercut_days_experiment"] = 0

    firewood_sources = {
        "q516_firewood_use_freq": "firewood_use_frequency_experiment",
        "q517_firewood_collect_purchase": "firewood_collect_purchase_experiment",
        "q517_a_firewood_collect_who": "firewood_collect_who_experiment",
        "q517_b_firewood_collect_freq": "firewood_collect_frequency_experiment",
    }
    for source, destination in firewood_sources.items():
        enhanced[destination] = enhanced[source].map(code_category)

    return enhanced


def load_training_data() -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Required modelling data was not found: {DATA_PATH}")
    data = pd.read_csv(DATA_PATH, low_memory=False)
    required = {TARGET, TARGET_MISSING_FLAG, "state"}
    missing = sorted(required - set(data.columns))
    if missing:
        raise KeyError(f"Required columns are missing from the modelling data: {missing}")

    observed = (
        data[TARGET_MISSING_FLAG]
        .astype(str)
        .str.strip()
        .str.lower()
        .eq("false")
    )
    modelling = data.loc[observed].copy()
    drop_columns = {
        col
        for col in modelling.columns
        if col in LEAKAGE_COLUMNS
        or col.startswith(("q610_", "q613_"))
        or col in SENSITIVE_COLUMNS
        or col in METADATA_COLUMNS
        or col == TARGET
    }
    target = pd.to_numeric(modelling[TARGET], errors="raise")
    features = modelling.drop(columns=list(drop_columns))

    required_sources = {
        "q314_pay_online_1",
        "q314_a_online_pay_ever_yn",
        "q314_b_online_pay_easy_diff",
        "q466_fridge_yn",
        "q467_fridge_no",
        "q468_fridge_most_yrs",
        "q469_fridge_most_bee_star",
        "q470_fridge_off_time",
        "survey_type_code",
        "q306_grid_rural_powercut_days_20",
        "q516_firewood_use_freq",
        "q517_firewood_collect_purchase",
        "q517_a_firewood_collect_who",
        "q517_b_firewood_collect_freq",
    }
    missing_sources = sorted(required_sources - set(features.columns))
    if missing_sources:
        raise KeyError(f"Required feature-engineering inputs are missing: {missing_sources}")

    enhanced = add_engineered_features(features)
    groups = modelling["state"].fillna("__missing_state__").astype(str)
    if enhanced.index.equals(target.index) is False:
        raise ValueError("Feature and target rows are not aligned.")
    return enhanced, target, groups


def split_training_data(
    features: pd.DataFrame,
    target: pd.Series,
    groups: pd.Series,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    x_train, _, y_train, _ = train_test_split(
        features,
        target,
        test_size=0.20,
        random_state=RANDOM_STATE,
    )
    return x_train, y_train, groups.loc[x_train.index]


def select_fold_features(
    train: pd.DataFrame, validation: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit the notebook's missingness/variance screening on each fold's train rows."""
    missing_rate = train.isna().mean()
    screened_columns = missing_rate[missing_rate <= 0.50].index
    screened = train.loc[:, screened_columns]
    unique_counts = screened.nunique(dropna=False)
    variable_columns = unique_counts[unique_counts > 1].index
    screened = screened.loc[:, variable_columns]
    dominant_share = screened.apply(
        lambda column: column.value_counts(normalize=True, dropna=False).max()
    )
    keep_columns = dominant_share[dominant_share < 0.99].index.tolist()
    if not keep_columns:
        raise ValueError("Fold feature screening removed every predictor.")
    return train.loc[:, keep_columns], validation.loc[:, keep_columns]


def identify_feature_types(
    frame: pd.DataFrame,
) -> tuple[list[str], list[str]]:
    categorical = [
        column
        for column in frame.columns
        if column in CATEGORICAL_COLUMNS
        or column in ENGINEERED_CATEGORICAL
        or pd.api.types.is_object_dtype(frame[column])
        or isinstance(frame[column].dtype, pd.CategoricalDtype)
        or pd.api.types.is_bool_dtype(frame[column])
    ]
    numeric = [column for column in frame.columns if column not in categorical]
    non_numeric = [
        column for column in numeric if not pd.api.types.is_numeric_dtype(frame[column])
    ]
    if non_numeric:
        raise TypeError(f"Unclassified non-numeric features: {non_numeric}")
    return categorical, numeric


def make_preprocessor(
    categorical: list[str], numeric: list[str]
) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "num",
                SimpleImputer(strategy="median", keep_empty_features=True),
                numeric,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        (
                            "imputer",
                            SimpleImputer(
                                strategy="most_frequent",
                                keep_empty_features=True,
                            ),
                        ),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical,
            ),
        ],
        sparse_threshold=0,
    )


def make_model(name: str) -> Any:
    if name.startswith("HGB"):
        return HistGradientBoostingRegressor(
            max_iter=400,
            learning_rate=0.05,
            max_leaf_nodes=31,
            min_samples_leaf=20,
            l2_regularization=5.0,
            random_state=RANDOM_STATE,
        )
    if name == "XGBoost engineered":
        return XGBRegressor(
            n_estimators=500,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=5.0,
            objective="reg:squarederror",
            tree_method="hist",
            n_jobs=4,
            random_state=RANDOM_STATE,
            verbosity=0,
        )
    if name == "CatBoost engineered":
        return CatBoostRegressor(
            iterations=600,
            depth=6,
            learning_rate=0.05,
            loss_function="RMSE",
            l2_leaf_reg=5.0,
            random_seed=RANDOM_STATE,
            thread_count=4,
            verbose=False,
            allow_writing_files=False,
        )
    raise ValueError(f"Unknown model: {name}")


def prepare_catboost_features(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    categorical: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = train.copy()
    validation = validation.copy()
    for column in categorical:
        train[column] = train[column].astype("string").fillna("__missing__").astype(str)
        validation[column] = (
            validation[column].astype("string").fillna("__missing__").astype(str)
        )
    return train, validation


def evaluate(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    groups: pd.Series,
    folds: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    x_train_base = x_train.drop(columns=list(ENGINEERED_COLUMNS))
    random_scheme = f"random_{folds}fold"
    grouped_scheme = f"state_group_{folds}fold"
    splitters: dict[str, Any] = {
        random_scheme: KFold(
            n_splits=folds, shuffle=True, random_state=RANDOM_STATE
        ),
    }
    if groups.nunique() < folds:
        raise ValueError(
            f"Need at least {folds} distinct states for grouped validation; "
            f"found {groups.nunique()}."
        )
    splitters[grouped_scheme] = GroupKFold(n_splits=folds)

    model_names = [
        "HGB baseline",
        "HGB engineered",
        "CatBoost engineered",
        "XGBoost engineered",
    ]
    fold_results: list[dict[str, Any]] = []
    prediction_results: list[dict[str, Any]] = []

    for scheme_name, splitter in splitters.items():
        splits = (
            splitter.split(x_train, y_train, groups=groups)
            if scheme_name == grouped_scheme
            else splitter.split(x_train, y_train)
        )
        for fold_number, (train_positions, valid_positions) in enumerate(splits, 1):
            raw_train = x_train_base.iloc[train_positions]
            raw_valid = x_train_base.iloc[valid_positions]
            y_fold_train = y_train.iloc[train_positions]
            y_fold_valid = y_train.iloc[valid_positions]
            base_train, base_valid = select_fold_features(raw_train, raw_valid)
            state_fold_valid = groups.iloc[valid_positions]

            for model_name in model_names:
                use_engineered = model_name != "HGB baseline"
                if use_engineered:
                    train_frame = base_train
                    valid_frame = base_valid
                    added_train = x_train.iloc[train_positions][
                        list(ENGINEERED_COLUMNS)
                    ]
                    added_valid = x_train.iloc[valid_positions][
                        list(ENGINEERED_COLUMNS)
                    ]
                    train_frame = train_frame.join(added_train)
                    valid_frame = valid_frame.join(added_valid)
                else:
                    train_frame = base_train
                    valid_frame = base_valid

                categorical, numeric = identify_feature_types(train_frame)
                model = make_model(model_name)
                if model_name == "CatBoost engineered":
                    train_frame, valid_frame = prepare_catboost_features(
                        train_frame, valid_frame, categorical
                    )
                    model.fit(
                        train_frame,
                        y_fold_train,
                        cat_features=categorical,
                    )
                    predictions = model.predict(valid_frame)
                else:
                    pipeline = Pipeline(
                        steps=[
                            (
                                "preprocessor",
                                make_preprocessor(categorical, numeric),
                            ),
                            ("model", model),
                        ]
                    )
                    pipeline.fit(train_frame, y_fold_train)
                    predictions = pipeline.predict(valid_frame)

                predictions = np.asarray(predictions, dtype=float)
                fold_results.append(
                    {
                        "scheme": scheme_name,
                        "model": model_name,
                        "fold": fold_number,
                        "n_train": len(train_positions),
                        "n_valid": len(valid_positions),
                        "features": train_frame.shape[1],
                        "MAE": mean_absolute_error(y_fold_valid, predictions),
                        "RMSE": np.sqrt(
                            mean_squared_error(y_fold_valid, predictions)
                        ),
                        "R2": r2_score(y_fold_valid, predictions),
                    }
                )
                for position, actual, prediction, state in zip(
                    valid_positions,
                    y_fold_valid.to_numpy(),
                    predictions,
                    state_fold_valid.to_numpy(),
                ):
                    prediction_results.append(
                        {
                            "scheme": scheme_name,
                            "model": model_name,
                            "fold": fold_number,
                            "row_index": x_train.index[position],
                            "state": state,
                            "actual": actual,
                            "prediction": prediction,
                            "residual": actual - prediction,
                            "absolute_error": abs(actual - prediction),
                        }
                    )
            print(f"Finished {scheme_name} fold {fold_number}/{folds}.")

    return pd.DataFrame(fold_results), pd.DataFrame(prediction_results)


def summarize(
    fold_results: pd.DataFrame, predictions: pd.DataFrame
) -> pd.DataFrame:
    summary = (
        fold_results.groupby(["scheme", "model"])
        .agg(
            MAE_mean=("MAE", "mean"),
            MAE_std=("MAE", "std"),
            RMSE_mean=("RMSE", "mean"),
            RMSE_std=("RMSE", "std"),
            R2_mean=("R2", "mean"),
            R2_std=("R2", "std"),
        )
        .reset_index()
    )
    pooled = (
        predictions.groupby(["scheme", "model"])
        .apply(
            lambda frame: pd.Series(
                {
                    "MAE_pooled": mean_absolute_error(
                        frame["actual"], frame["prediction"]
                    ),
                    "RMSE_pooled": np.sqrt(
                        mean_squared_error(frame["actual"], frame["prediction"])
                    ),
                    "R2_pooled": r2_score(frame["actual"], frame["prediction"]),
                }
            ),
            include_groups=False,
        )
        .reset_index()
    )
    return summary.merge(pooled, on=["scheme", "model"])


def save_error_segments(predictions: pd.DataFrame, random_scheme: str) -> None:
    random_predictions = predictions.loc[
        predictions["scheme"] == random_scheme
    ].copy()
    random_predictions["bill_range"] = pd.qcut(
        random_predictions["actual"],
        q=5,
        duplicates="drop",
    )
    bill_segments = (
        random_predictions.groupby(
            ["model", "bill_range"], observed=True
        )
        .agg(
            households=("actual", "size"),
            actual_mean=("actual", "mean"),
            prediction_mean=("prediction", "mean"),
            MAE=("absolute_error", "mean"),
            mean_residual=("residual", "mean"),
        )
        .reset_index()
    )
    state_segments = (
        random_predictions.groupby(["model", "state"], observed=True)
        .agg(
            households=("actual", "size"),
            MAE=("absolute_error", "mean"),
            mean_residual=("residual", "mean"),
        )
        .reset_index()
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    bill_segments.to_csv(OUTPUT_DIR / "model_benchmark_error_by_bill_range.csv", index=False)
    state_segments.to_csv(OUTPUT_DIR / "model_benchmark_error_by_state.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()
    if args.folds < 2:
        raise ValueError("--folds must be at least 2.")

    features, target, groups = load_training_data()
    x_train, y_train, train_groups = split_training_data(features, target, groups)
    fold_results, predictions = evaluate(
        x_train, y_train, train_groups, args.folds
    )
    summary = summarize(fold_results, predictions)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fold_results.to_csv(OUTPUT_DIR / "model_benchmark_folds.csv", index=False)
    predictions.to_csv(OUTPUT_DIR / "model_benchmark_oof_predictions.csv", index=False)
    summary.to_csv(OUTPUT_DIR / "model_benchmark_summary.csv", index=False)
    random_scheme = f"random_{args.folds}fold"
    grouped_scheme = f"state_group_{args.folds}fold"
    save_error_segments(predictions, random_scheme)

    metadata = {
        "target": TARGET,
        "target_interpretation": "Average monthly electricity bill, not electricity use in kWh.",
        "training_households": int(len(y_train)),
        "folds_per_scheme": args.folds,
        "validation_schemes": [
            random_scheme,
            grouped_scheme,
        ],
        "test_split_used": False,
        "preprocessing": (
            "Missingness and variance-based feature screening, imputation, and "
            "one-hot categories are fit using each training fold only."
        ),
        "feature_engineering": (
            "The same ten rule-based features as the existing notebook; no target "
            "values are used to construct them."
        ),
        "model_settings": (
            "Existing HGB settings are reused. CatBoost and XGBoost use fixed, "
            "moderate tree settings; no hyperparameter search was performed."
        ),
    }
    (OUTPUT_DIR / "model_benchmark_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print("\nCross-validation summary:")
    print(summary.sort_values(["scheme", "MAE_pooled"]).round(3).to_string(index=False))
    print(f"\nResults saved under {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
