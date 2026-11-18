"""Streamlit dashboard for survey-based household energy insights."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from ml.dashboard_models import (
    BILL_TARGET,
    CONSUMPTION_TARGET,
    FEATURES,
    observed_bill_mask,
    OUTPUT_DIR,
)


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "output" / "analysis_ready_data.csv"
METRICS_PATH = OUTPUT_DIR / "dashboard_model_metrics.json"

st.set_page_config(
    page_title="CEEW Household Energy Intelligence",
    page_icon="⚡",
    layout="wide",
)


@st.cache_data
def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Analysis-ready data not found: {DATA_PATH}")
    return pd.read_csv(DATA_PATH, low_memory=False)


@st.cache_resource
def load_artifact(path: str) -> Any:
    artifact_path = Path(path)
    if not artifact_path.exists():
        raise FileNotFoundError(
            f"Model artifact not found: {artifact_path}. "
            "Run `python ml/dashboard_models.py` first."
        )
    return joblib.load(artifact_path)


@st.cache_data
def load_metrics() -> dict[str, Any]:
    if not METRICS_PATH.exists():
        raise FileNotFoundError(
            f"Model metrics not found: {METRICS_PATH}. "
            "Run `python ml/dashboard_models.py` first."
        )
    return json.loads(METRICS_PATH.read_text(encoding="utf-8"))


def show_validation_metrics(metrics: dict[str, float], unit: str) -> None:
    first, second, third = st.columns(3)
    first.metric("R²", f"{metrics['R2']:.3f}")
    second.metric("MAE", f"{unit}{metrics['MAE']:,.1f}")
    third.metric("RMSE", f"{unit}{metrics['RMSE']:,.1f}")


def household_profile(data: pd.DataFrame) -> pd.DataFrame:
    state_options = sorted(data["state"].dropna().astype(str).unique())
    with st.form("household_profile"):
        st.subheader("Enter a household profile")
        state = st.selectbox("State", state_options)
        left, right = st.columns(2)
        with left:
            household_size = st.number_input(
                "Household members", min_value=1, max_value=30, value=4
            )
            age = st.number_input("Respondent age", min_value=18, max_value=100, value=40)
            monthly_expenditure = st.number_input(
                "Monthly household expenditure (₹)",
                min_value=0,
                max_value=500000,
                value=10000,
                step=500,
            )
            grid_hours = st.number_input(
                "Electricity supply hours per day", min_value=0, max_value=24, value=20
            )
            grid_connected = st.checkbox("Connected to grid", value=True)
            has_meter = st.checkbox("Has an electricity meter", value=True)
            has_fridge = st.checkbox("Has a refrigerator", value=True)
        with right:
            fridge_count = st.number_input(
                "Refrigerators", min_value=0, max_value=10, value=1
            )
            fan_count = st.number_input("Ceiling fans", min_value=0, max_value=30, value=3)
            tv_count = st.number_input("Televisions", min_value=0, max_value=10, value=1)
            led_count = st.number_input("LED bulbs", min_value=0, max_value=100, value=4)
            has_ac = st.checkbox("Has an air conditioner", value=False)
            uses_png = st.checkbox("Uses piped natural gas", value=False)
            uses_lpg = st.checkbox("Uses LPG", value=True)
        submitted = st.form_submit_button("Estimate monthly bill")

    profile = pd.DataFrame(
        [
            {
                "state": state,
                "respondent_age": age,
                "household_size": household_size,
                "monthly_expenditure": monthly_expenditure,
                "q302_grid_hrs_no": grid_hours,
                "q467_fridge_no": fridge_count if has_fridge else 0,
                "q410_ceiling_fan_no": fan_count,
                "q460_tv_no": tv_count,
                "q405_c_led_bulb_no": led_count,
                "q301_grid_yn": int(grid_connected),
                "q312_grid_elec_meter_yn": int(has_meter),
                "q466_fridge_yn": int(has_fridge),
                "q427_ac_yn": int(has_ac),
                "q501_png_use_yn": int(uses_png),
                "q502_lpg_use_yn": int(uses_lpg),
            }
        ],
        columns=FEATURES,
    )
    return profile if submitted else pd.DataFrame()


def show_energy_intelligence(data: pd.DataFrame) -> None:
    st.subheader("Household comparisons")
    states = sorted(data["state"].dropna().astype(str).unique())
    selected_state = st.selectbox("Comparison state", states, key="comparison_state")
    selected_size = st.selectbox(
        "Household size",
        ["Any", "1–3", "4–6", "7+"],
        key="comparison_household_size",
    )

    peers = data.loc[
        data["state"].astype("string").eq(selected_state)
        & observed_bill_mask(data)
    ].copy()
    if selected_size != "Any":
        lower, upper = {
            "1–3": (1, 3),
            "4–6": (4, 6),
            "7+": (7, np.inf),
        }[selected_size]
        peers = peers.loc[peers["household_size"].between(lower, upper)]

    if peers.empty:
        st.info("No households match this comparison group.")
        return
    q25, median, q75 = peers[BILL_TARGET].quantile([0.25, 0.5, 0.75])
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Households in group", f"{len(peers):,}")
    col2.metric("Median monthly bill", f"₹{median:,.0f}")
    col3.metric("25th percentile", f"₹{q25:,.0f}")
    col4.metric("75th percentile", f"₹{q75:,.0f}")
    st.caption(
        "Unweighted survey-sample comparisons; these are not national estimates. "
        "A high value is a prompt to check bill period, tariff, and meter readings—not "
        "evidence of waste or wrongdoing."
    )
    st.bar_chart(
        peers.groupby("bill_group", observed=True)[BILL_TARGET].median().rename(
            "Median bill (₹)"
        )
    )


def main() -> None:
    st.title("CEEW Household Energy Intelligence")
    st.caption(
        "Exploration of the 2019–2020 India Residential Energy Survey. "
        "Not a live billing service or a current household forecast."
    )
    data = load_data()
    metrics = load_metrics()

    total, states, rows_with_units = st.columns(3)
    total.metric("Survey households", f"{len(data):,}")
    states.metric("States represented", f"{data['state'].nunique():,}")
    rows_with_units.metric(
        "Households with billed units",
        f"{metrics['consumption_pilot']['observed_rows']:,}",
    )

    prediction_tab, intelligence_tab, methodology_tab = st.tabs(
        ["Predictions", "Energy intelligence", "Validation & limitations"]
    )
    with prediction_tab:
        profile = household_profile(data)
        if not profile.empty:
            bill_model = load_artifact(str(OUTPUT_DIR / "dashboard_bill_model.pkl"))
            bill_estimate = float(bill_model.predict(profile)[0])
            st.subheader("Model estimates")
            st.metric("Estimated monthly bill", f"₹{bill_estimate:,.0f}")
            random_mae = metrics["models"][BILL_TARGET]["cross_validation"][
                "random_5fold"
            ]["MAE"]
            state_mae = metrics["models"][BILL_TARGET]["cross_validation"][
                "state_held_out_5fold"
            ]["MAE"]
            st.caption(
                f"Profile-model cross-validation MAE: ₹{random_mae:,.0f} on shuffled "
                f"folds and ₹{state_mae:,.0f} on held-out states. These are typical "
                "validation errors, not an individual prediction interval."
            )
            state_peers = data.loc[
                data["state"].astype("string").eq(profile.at[0, "state"])
                & observed_bill_mask(data)
            ]
            peer_count = len(state_peers)
            if peer_count >= 20:
                state_q75 = float(state_peers[BILL_TARGET].quantile(0.75))
                if bill_estimate > state_q75:
                    st.info(
                        "This estimate is above the 75th percentile of the state's "
                        "unweighted survey bills. Check bill period, tariff, and meter "
                        "readings; this is not evidence of waste or wrongdoing."
                    )
            st.caption(
                "Estimates are for a profile entered by the user. They are not utility "
                "quotes, causal energy-saving estimates, or guaranteed individual outcomes."
            )

    with intelligence_tab:
        show_energy_intelligence(data)

    with methodology_tab:
        st.subheader("Bill profile model")
        bill_results = metrics["models"][BILL_TARGET]["cross_validation"]
        st.write("Shuffled five-fold validation:")
        show_validation_metrics(bill_results["random_5fold"], "₹")
        st.write("State-held-out five-fold validation:")
        show_validation_metrics(bill_results["state_held_out_5fold"], "₹")

        st.subheader("Billed-units pilot model")
        consumption_results = metrics["models"][CONSUMPTION_TARGET][
            "cross_validation"
        ]
        st.write("Shuffled five-fold validation:")
        show_validation_metrics(consumption_results["random_5fold"], "")
        st.write("State-held-out five-fold validation:")
        show_validation_metrics(consumption_results["state_held_out_5fold"], "")
        st.dataframe(
            pd.DataFrame.from_dict(
                metrics["consumption_pilot"]["coverage_by_state"],
                orient="index",
            ).sort_values("coverage"),
            use_container_width=True,
        )
        st.warning(metrics["consumption_pilot"]["warning"])
        st.info(
            "Consumption prediction is intentionally disabled in the interactive form "
            "because the billed-units target is incomplete and performs poorly for "
            "held-out states. The dashboard shows target coverage and validation results "
            "instead of presenting an unreliable estimate."
        )
        st.subheader("Global bill-model explainability")
        importance = pd.DataFrame(
            metrics["models"][BILL_TARGET]["global_permutation_importance"]
        ).head(10)
        st.bar_chart(importance.set_index("feature")["mean_r2_decrease"])
        st.caption(
            "Mean change in held-out-state R² when a feature is shuffled. This is a "
            "global predictive association, not a causal explanation for an individual bill."
        )
        st.caption(
            "Metrics describe this survey sample only. State-held-out scores are the more "
            "conservative check for applying a model in a new state. Comparisons are "
            "unweighted until the survey weighting scheme has been verified."
        )


if __name__ == "__main__":
    main()
