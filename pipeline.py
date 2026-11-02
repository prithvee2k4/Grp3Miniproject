"""CEEW India Residential Energy Survey 2020 - ETL pipeline.

Run:  python pipeline.py
"""
import time

from etl import config, log
from etl import extract_step, member2_structural_cleaning, member3_missing_invalid
from etl import member4_features, member5_reports_load


def extract():
    """Read the raw CSV, show its size, validate it and save a copy in raw_data/."""
    return extract_step.run_extract()


def transform(df):
    """Clean and transform: Member 2 -> Member 3 -> Member 4."""
    print("\n--- TRANSFORM ---")
    df = member2_structural_cleaning.run(df)
    df = member3_missing_invalid.run(df)
    df = member4_features.run(df)
    log.step(f"[Transform] finished: {df.shape[0]:,} rows x {df.shape[1]} columns")
    return df


def generate_reports(df):
    """Build the report tables (Member 5)."""
    print("\n--- REPORTS ---")
    return member5_reports_load.generate_reports(df)


def load(df, reports):
    """Save reports to reports/ and the final dataset to output/."""
    print("\n--- LOAD ---")
    member5_reports_load.save_reports(reports, df)


def run_etl_pipeline():
    """Extract -> Transform -> Reports -> Load."""
    start = time.time()
    log.reset()
    print("--- EXTRACT ---")
    raw = extract()
    clean = transform(raw)
    reports = generate_reports(clean)
    load(clean, reports)
    print(f"\nPIPELINE FINISHED in {time.time() - start:.1f}s: "
          f"raw {raw.shape[0]:,} x {raw.shape[1]} -> clean {clean.shape[0]:,} x {clean.shape[1]}")
    return clean


if __name__ == "__main__":
    run_etl_pipeline()
