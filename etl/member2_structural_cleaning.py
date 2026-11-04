"""Member 2 - Structural cleaning.

Transformations 1-6: rename columns, remove duplicates, drop empty columns,
standardise text, convert dates and change data types.
"""

import numpy as np
import pandas as pd

from etl import config, log

IMPLEMENTED = True

TITLE_CASE_COLUMNS = ["state", "district", "village_ward"]
UPPER_CASE_COLUMNS = ["state_abbr"]
MAX_CATEGORY_VALUES = 50
ID_COLUMNS = ["hhid"]


def rename_columns(df):
    """Transformation 1: give the important survey codes readable names."""
    mapping = {
        old: new for old, new in config.RENAME_MAP.items()
        if old in df.columns
    }
    df = df.rename(columns=mapping)
    log.step(f"[Member 2] Renamed {len(mapping)} columns to readable names")
    return df


def remove_duplicates(df):
    """Transformation 2: keep one row per household."""
    before = len(df)
    df = df.drop_duplicates()

    if "hhid" in df.columns:
        df = df.drop_duplicates(subset="hhid")

    df = df.reset_index(drop=True)
    log.step(f"[Member 2] Removed {before - len(df)} duplicate rows")
    return df


def drop_empty_columns(df):
    """Transformation 3: drop columns that are more than 90% empty."""
    share_missing = df.isna().mean()

    to_drop = [
        col for col in df.columns
        if share_missing[col] > config.MAX_MISSING_SHARE
        and col not in config.KEY_NUMERIC
    ]

    df = df.drop(columns=to_drop)

    log.step(
        f"[Member 2] Dropped {len(to_drop)} columns that are more than "
        f"{config.MAX_MISSING_SHARE:.0%} empty"
    )
    return df


def _is_text(series):
    """Check whether a column contains text data."""
    return (
        pd.api.types.is_string_dtype(series)
        or pd.api.types.is_object_dtype(series)
    )


def standardize_text(df):
    """Transformation 4: trim spaces and standardise text."""
    df = df.copy()

    for col in df.columns:
        if not _is_text(df[col]):
            continue

        text = df[col].str.strip().str.replace(
            r"\s+", " ", regex=True
        )

        if col in UPPER_CASE_COLUMNS:
            text = text.str.upper()
        elif col in TITLE_CASE_COLUMNS:
            text = text.str.title()

        df[col] = text

    log.step(
        "[Member 2] Standardised text columns "
        "(trimmed spaces, fixed letter case)"
    )
    return df


def convert_dates(df):
    """Transformation 5: convert dates and extract interview hour."""
    df = df.copy()

    if "interview_date" in df.columns:
        text = df["interview_date"].astype("string").str.strip()

        slash = pd.to_datetime(
            text, format="%m/%d/%Y", errors="coerce"
        )
        dash = pd.to_datetime(
            text, format="%m-%d-%Y", errors="coerce"
        )

        df["interview_date"] = slash.fillna(dash)

        failed = int(df["interview_date"].isna().sum())

        log.step(
            f"[Member 2] Converted interview_date to dates "
            f"({failed} could not be read)"
        )

    if "interview_start_time" in df.columns:
        text = df["interview_start_time"].astype(
            "string"
        ).str.strip()

        times = pd.to_datetime(
            text, format="%I:%M %p", errors="coerce"
        )

        df["interview_start_hour"] = times.dt.hour.astype("Int8")
        log.step("[Member 2] Added interview_start_hour")

    return df


def change_dtypes(df):
    """Transformation 6: use smaller, safer data types."""
    df = df.copy()

    before = df.memory_usage(deep=True).sum() / 1e6

    for col in df.columns:
        series = df[col]

        if col in ID_COLUMNS:
            continue

        if isinstance(series.dtype, np.dtype) and series.dtype.kind == "i":
            smaller = pd.to_numeric(series, downcast="integer")

            # Keep at least int16 for safer arithmetic.
            if smaller.dtype.itemsize < 2:
                df[col] = smaller.astype("int16")
            else:
                df[col] = smaller

        elif _is_text(series) and series.nunique() <= MAX_CATEGORY_VALUES:
            df[col] = series.astype("category")

    after = df.memory_usage(deep=True).sum() / 1e6

    log.step(
        f"[Member 2] Changed data types: "
        f"memory {before:.1f} MB -> {after:.1f} MB"
    )

    return df


def run(df):
    """Run all structural cleaning steps in order."""
    df = rename_columns(df)
    df = remove_duplicates(df)
    df = drop_empty_columns(df)
    df = standardize_text(df)
    df = convert_dates(df)
    df = change_dtypes(df)

    log.step(
        f"[Member 2] finished: "
        f"{df.shape[0]:,} rows x {df.shape[1]} columns"
    )

    return df
