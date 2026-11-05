
"""Member 4 - Features, categories, ranking and sorting.

Transformations 11-14: calculated columns, categories, ranking and sorting.
"""
import numpy as np
import pandas as pd

from etl import log

IMPLEMENTED = True


def add_calculated_columns(df):
    """Transformation 11: create per-member and ratio columns."""
    df = df.copy()

    size = df["household_size"].replace(0, np.nan)

    df["elec_bill_per_member"] = (
        df["avg_monthly_elec_bill"] / size
    ).round(2)

    df["expenditure_per_member"] = (
        df["monthly_expenditure"] / size
    ).round(2)

    expenditure = df["monthly_expenditure"].replace(0, np.nan)

    df["bill_share_of_expenditure_pct"] = (
        df["avg_monthly_elec_bill"] / expenditure * 100
    ).round(2)

    if "bedrooms" in df.columns:
        bedrooms = df["bedrooms"].replace(0, np.nan)
        df["people_per_bedroom"] = (size / bedrooms).round(2)

    log.step("[Member 4] Added calculated columns")

    return df


def run(df):
    """Run all Member 4 transformations."""
    df = add_calculated_columns(df)
    df = add_categories(df)
    df = add_rankings(df)
    df = sort_records(df)
    return df


def add_categories(df):
    """Transformation 12: create age, household and spending groups."""
    df = df.copy()

    inf = np.inf

    df["age_group"] = pd.cut(
        df["respondent_age"],
        bins=[18, 30, 45, 60, inf],
        right=False,
        labels=["18-29", "30-44", "45-59", "60+"]
    )

    df["household_size_group"] = pd.cut(
        df["household_size"],
        bins=[1, 4, 7, inf],
        right=False,
        labels=["Small (1-3)", "Medium (4-6)", "Large (7+)"]
    )

    df["expenditure_group"] = pd.cut(
        df["monthly_expenditure"],
        bins=[0, 5000, 10000, 20000, inf],
        right=False,
        labels=[
            "Low (<5k)",
            "Medium (5k-10k)",
            "High (10k-20k)",
            "Very high (20k+)"
        ]
    )

    df["bill_group"] = pd.cut(
        df["avg_monthly_elec_bill"],
        bins=[0, 300, 700, 1500, inf],
        right=False,
        labels=[
            "Low (<300)",
            "Medium (300-699)",
            "High (700-1499)",
            "Very high (1500+)"
        ]
    )

    log.step(
        "[Member 4] Created age, household size, expenditure and bill groups"
    )

    return df


def add_rankings(df):
    """Transformation 13: rank states and households by electricity bill."""
    df = df.copy()

    state_avg = (
        df.groupby("state", observed=True)["avg_monthly_elec_bill"]
        .mean()
    )

    state_rank = state_avg.rank(
        ascending=False,
        method="dense"
    )

    df["state_bill_rank"] = (
        df["state"].map(state_rank).astype("Int64")
    )

    df["household_bill_rank"] = (
        df["avg_monthly_elec_bill"]
        .rank(ascending=False, method="min")
        .astype("Int64")
    )

    log.step("[Member 4] Added state and household electricity bill rankings")

    return df


def sort_records(df):
    """Transformation 14: sort by state, district and household ID."""
    df = df.copy()

    df = df.sort_values(
        by=["state", "district", "hhid"]
    ).reset_index(drop=True)

    log.step("[Member 4] Sorted records by state, district and hhid")

    return df
