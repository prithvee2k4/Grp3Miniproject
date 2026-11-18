"""Train the lightweight models used by the interactive dashboard."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "output" / "analysis_ready_data.csv"
OUTPUT_DIR = ROOT / "output" / "ml_preprocessed"
BILL_TARGET = "avg_monthly_elec_bill"
CONSUMPTION_TARGET = "q610_b_billed_units"
MISSING_FLAG = "avg_monthly_elec_bill_was_missing"
RANDOM_STATE = 42
FOLDS = 5

NUMERIC_FEATURES = [
    "respondent_age",
    "household_size",
    "monthly_expenditure",
    "q302_grid_hrs_no",
    "q467_fridge_no",
    "q410_ceiling_fan_no",
    "q460_tv_no",
    "q405_c_led_bulb_no",
]
CATEGORICAL_FEATURES = [
    "state",
    "q301_grid_yn",
    "q312_grid_elec_meter_yn",
    "q466_fridge_yn",
    "q427_ac_yn",
    "q501_png_use_yn",
    "q502_lpg_use_yn",
]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def observed_bill_mask(data: pd.DataFrame) -> pd.Series:
    if MISSING_FLAG not in data:
        raise KeyError(f"Required column is missing: {MISSING_FLAG}")
    flag = data[MISSING_FLAG].astype("string").str.strip().str.lower()
    return flag.eq("false").fillna(False)


def prepare_training_data(
    data: pd.DataFrame, target_name: str
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Return user-available features, observed target values, and state groups."""
    if target_name == BILL_TARGET:
        required_columns = set(FEATURES) | {target_name, MISSING_FLAG, "state"}
    elif target_name != CONSUMPTION_TARGET:
        raise ValueError(f"Unsupported dashboard target: {target_name}")
    else:
        required_columns = set(FEATURES) | {target_name, "state"}
    missing_columns = sorted(required_columns - set(data.columns))
    if missing_columns:
        raise KeyError(f"Required columns are missing: {missing_columns}")

    target_values = pd.to_numeric(data[target_name], errors="coerce")
    observed = target_values.notna() & np.isfinite(target_values)
    if target_name == BILL_TARGET:
        observed &= observed_bill_mask(data)

    features = data.loc[observed, FEATURES].copy()
    for column in NUMERIC_FEATURES:
        features[column] = pd.to_numeric(features[column], errors="coerce")
    for column in CATEGORICAL_FEATURES:
        values = features[column].astype(object)
        features[column] = values.where(pd.notna(values), np.nan)

    target = target_values.loc[observed].astype(float)
    groups = (
        data.loc[observed, "state"]
        .astype("string")
        .fillna("__missing_state__")
        .astype(str)
    )
    return features, target, groups


def make_model_pipeline() -> Pipeline:
    numeric = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median", keep_empty_features=True))]
    )
    categorical = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric, NUMERIC_FEATURES),
            ("categorical", categorical, CATEGORICAL_FEATURES),
        ],
        sparse_threshold=0,
    )
    model = HistGradientBoostingRegressor(
        max_iter=250,
        learning_rate=0.05,
        max_leaf_nodes=15,
        min_samples_leaf=20,
        l2_regularization=5.0,
        random_state=RANDOM_STATE,
    )
    return Pipeline([("preprocessor", preprocessor), ("model", model)])


def evaluate_model(
    features: pd.DataFrame,
    target: pd.Series,
    groups: pd.Series,
) -> dict[str, dict[str, float]]:
    """Evaluate profile models with random and state-held-out out-of-fold predictions."""
    if groups.nunique() < FOLDS:
        raise ValueError(
            f"Need at least {FOLDS} states for grouped validation; found {groups.nunique()}."
        )

    splitters: dict[str, Any] = {
        "random_5fold": KFold(n_splits=FOLDS, shuffle=True, random_state=RANDOM_STATE),
        "state_held_out_5fold": GroupKFold(n_splits=FOLDS),
    }
    results: dict[str, dict[str, float]] = {}
    for name, splitter in splitters.items():
        predictions = np.full(len(target), np.nan)
        splits = (
            splitter.split(features, target, groups=groups)
            if name == "state_held_out_5fold"
            else splitter.split(features, target)
        )
        for train_indices, validation_indices in splits:
            model = make_model_pipeline()
            model.fit(features.iloc[train_indices], target.iloc[train_indices])
            predictions[validation_indices] = model.predict(
                features.iloc[validation_indices]
            )
        if not np.isfinite(predictions).all():
            raise RuntimeError(f"Validation left predictions missing for {name}.")
        results[name] = {
            "MAE": float(mean_absolute_error(target, predictions)),
            "RMSE": float(np.sqrt(mean_squared_error(target, predictions))),
            "R2": float(r2_score(target, predictions)),
        }
    return results


def evaluate_grouped_permutation_importance(
    features: pd.DataFrame,
    target: pd.Series,
    groups: pd.Series,
) -> list[dict[str, float | str]]:
    """Estimate global, non-causal feature importance on held-out states."""
    splitter = GroupKFold(n_splits=FOLDS)
    fold_importances = []
    for train_indices, validation_indices in splitter.split(
        features, target, groups=groups
    ):
        model = make_model_pipeline()
        model.fit(features.iloc[train_indices], target.iloc[train_indices])
        result = permutation_importance(
            model,
            features.iloc[validation_indices],
            target.iloc[validation_indices],
            scoring="r2",
            n_repeats=3,
            random_state=RANDOM_STATE,
            n_jobs=1,
        )
        fold_importances.append(result.importances_mean)

    mean_importance = np.mean(fold_importances, axis=0)
    return [
        {"feature": str(feature), "mean_r2_decrease": float(importance)}
        for feature, importance in sorted(
            zip(features.columns, mean_importance),
            key=lambda item: item[1],
            reverse=True,
        )
    ]


def train_dashboard_models(data: pd.DataFrame | None = None) -> dict[str, Any]:
    """Evaluate and persist the bill model and the explicitly experimental units model."""
    if data is None:
        if not DATA_PATH.exists():
            raise FileNotFoundError(f"Required modelling data was not found: {DATA_PATH}")
        data = pd.read_csv(DATA_PATH, low_memory=False)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model_results: dict[str, Any] = {}
    for target_name in (BILL_TARGET, CONSUMPTION_TARGET):
        features, target, groups = prepare_training_data(data, target_name)
        metrics = evaluate_model(features, target, groups)
        model_results[target_name] = {
            "training_rows": int(len(target)),
            "features": FEATURES,
            "cross_validation": metrics,
        }
        if target_name == BILL_TARGET:
            model_results[target_name]["global_permutation_importance"] = (
                evaluate_grouped_permutation_importance(features, target, groups)
            )
            final_model = make_model_pipeline()
            final_model.fit(features, target)
            artifact_name = "dashboard_bill_model.pkl"
            joblib.dump(final_model, OUTPUT_DIR / artifact_name)
            model_results[target_name]["artifact"] = artifact_name

    observed_units = pd.to_numeric(data[CONSUMPTION_TARGET], errors="coerce").notna()
    coverage_data = pd.DataFrame(
        {"state": data["state"], "_units_observed": observed_units}
    )
    state_rows = coverage_data.groupby("state", dropna=False, observed=True)
    coverage_by_state = {}
    for state, state_frame in state_rows:
        state_name = "__missing_state__" if pd.isna(state) else str(state)
        coverage_by_state[state_name] = {
            "households": int(len(state_frame)),
            "units_observed": int(state_frame["_units_observed"].sum()),
            "coverage": float(state_frame["_units_observed"].mean()),
        }

    report = {
        "target_scope": {
            "bill": "Average monthly electricity bill in rupees.",
            "consumption": (
                "q610_b_billed_units: units recorded on an observed electricity bill; "
                "the codebook must confirm the unit and billing-period interpretation."
            ),
        },
        "consumption_pilot": {
            "status": "exploratory_only",
            "prediction_enabled": False,
            "observed_rows": int(observed_units.sum()),
            "total_rows": int(len(data)),
            "coverage": float(observed_units.mean()),
            "coverage_by_state": coverage_by_state,
            "warning": (
                "Billed-unit observations are sparse and highly uneven across states. "
                "State-held-out validation is required before enabling predictions; do not "
                "interpret the target as monthly kWh without verifying the survey codebook."
            ),
        },
        "models": model_results,
        "validation": (
            "Five-fold shuffled and state-held-out cross-validation; preprocessing is "
            "fitted within each fold. Dashboard models use only the documented profile fields."
        ),
    }
    metrics_path = OUTPUT_DIR / "dashboard_model_metrics.json"
    metrics_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    result = train_dashboard_models()
    for target_name, model_info in result["models"].items():
        print(f"\n{target_name}: {model_info['training_rows']} training rows")
        for scheme, scores in model_info["cross_validation"].items():
            print(f"  {scheme}: {scores}")
    print(
        "\nConsumption pilot coverage: "
        f"{result['consumption_pilot']['observed_rows']}/"
        f"{result['consumption_pilot']['total_rows']} "
        f"({result['consumption_pilot']['coverage']:.1%})"
    )
