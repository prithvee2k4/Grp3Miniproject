
"""Member 3 - Missing and invalid values, outliers.

Transformations 7-10: replace "don't know" codes, validate values,
cap outliers and fill missing values in the key columns.
"""
import pandas as pd

from etl import config, log

IMPLEMENTED = True

CODED_COLUMN_MAX_VALUES = 15

AGE_GROUP_COLUMNS = [
    "members_under_18",
    "members_18_40",
    "members_41_60",
    "members_over_60",
]


def replace_dont_know_codes(df):
    """Transformation 7: replace don't-know codes with missing values."""
    df = df.copy()
    codes = list(config.DONT_KNOW_CODES)
    replaced = 0

    for col in df.select_dtypes(include="number").columns:
        if col == "hhid":
            continue

        if df[col].nunique(dropna=True) > CODED_COLUMN_MAX_VALUES:
            continue

        is_code = df[col].isin(codes)
        count = int(is_code.sum())

        if count:
            df[col] = df[col].mask(is_code)
            replaced += count

    log.step(
        f"[Member 3] Replaced {replaced:,} don't-know codes "
        "with missing values"
    )
    return df


def validate_values(df):
    """Transformation 8: blank out impossible values."""
    df = df.copy()
    low_age, high_age = config.VALID_AGE
    problems = {}

    def blank(col, bad):
        if col not in df.columns:
            return

        bad = bad.fillna(False)
        count = int(bad.sum())

        if count:
            problems[col] = count
            df[col] = df[col].mask(bad)

    if "respondent_age" in df.columns:
        age = df["respondent_age"]
        blank("respondent_age", (age < low_age) | (age > high_age))

    if "household_size" in df.columns:
        blank("household_size", df["household_size"] < 1)

    spending_columns = [
        "land_area_sqft",
        "monthly_expenditure",
        "avg_monthly_elec_bill",
    ]

    for col in spending_columns:
        if col in df.columns:
            blank(col, df[col] <= 0)

    groups = [
        col for col in AGE_GROUP_COLUMNS
        if col in df.columns
    ]

    if (
        len(groups) == len(AGE_GROUP_COLUMNS)
        and "household_size" in df.columns
    ):
        complete = df[groups].notna().all(axis=1)
        complete &= df["household_size"].notna()

        mismatch = complete & (
            df[groups].sum(axis=1) != df["household_size"]
        )

        count = int(mismatch.sum())

        if count:
            problems["age_group_counts"] = count

            for col in groups:
                df[col] = df[col].mask(mismatch)

    log.step(
        f"[Member 3] Validation blanked invalid values: "
        f"{problems or 'none found'}"
    )
    return df


def cap_outliers(df):
    """Transformation 9: cap values at the configured percentiles."""
    df = df.copy()
    low, high = config.OUTLIER_PERCENTILES

    for col in config.KEY_NUMERIC:
        if col not in df.columns:
            continue

        if not pd.api.types.is_numeric_dtype(df[col]):
            continue

        if df[col].notna().sum() == 0:
            continue

        lower, upper = df[col].quantile([low, high])

        if pd.isna(lower) or pd.isna(upper):
            continue

        capped = int(
            ((df[col] < lower) | (df[col] > upper)).sum()
        )

        df[col] = df[col].clip(lower=lower, upper=upper)

        log.step(
            f"[Member 3] Capped {capped:,} outliers in {col} "
            f"to [{lower:.0f}, {upper:.0f}]"
        )

    return df


def fill_missing_values(df):
    """Transformation 10: fill key values using state medians."""
    df = df.copy()

    for col in config.KEY_NUMERIC:
        if col not in df.columns:
            continue

        if not pd.api.types.is_numeric_dtype(df[col]):
            continue

        was_missing = df[col].isna()
        df[col + "_was_missing"] = was_missing

        if "state" in df.columns:
            state_median = (
                df.groupby("state", observed=True)[col]
                .transform("median")
            )
            df[col] = df[col].fillna(state_median)

        overall_median = df[col].median()

        if pd.notna(overall_median):
            df[col] = df[col].fillna(overall_median)

        log.step(
            f"[Member 3] Filled "
            f"{int(was_missing.sum()):,} missing values in {col}"
        )

    return df


def run(df):
    """Run all Member 3 transformations in order."""
    df = replace_dont_know_codes(df)
    df = validate_values(df)
    df = cap_outliers(df)
    df = fill_missing_values(df)

    log.step(
        f"[Member 3] finished: "
        f"{df.shape[0]:,} rows x {df.shape[1]} columns"
    )

    return df
