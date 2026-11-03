
"""Member 1 - Extract and validate the raw CEEW IRES 2020 dataset."""

import pandas as pd

from etl import config, log


IMPLEMENTED = True


def validate_raw(df):
    """Validate the extracted raw dataset."""

    if df.empty:
        raise ValueError("The raw dataset is empty.")

    missing_columns = [
        column for column in config.REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Required columns missing from raw dataset: {missing_columns}"
        )

    log.step(
        f"[Extract] Validation successful: "
        f"{df.shape[0]:,} rows x {df.shape[1]} columns"
    )

    return df


def run_extract(source=None, save_copy=True):
    """Read, validate and optionally save a copy of the raw dataset."""

    file_path = source if source is not None else config.SOURCE_FILE

    log.step(f"[Extract] Reading dataset: {file_path}")

    df = pd.read_csv(file_path, low_memory=False)

    validate_raw(df)

    if save_copy:
        config.RAW_COPY.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(config.RAW_COPY, index=False)

        log.step(f"[Extract] Raw copy saved: {config.RAW_COPY}")

    return df
