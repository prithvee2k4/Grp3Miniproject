
"""Member 5 - Reports and load.

Generate summary reports and save the analysis-ready dataset.
"""

import pandas as pd

from etl import config, log


IMPLEMENTED = True


def generate_reports(df):
    """Generate all required report DataFrames."""

    print("[Member 5] Generating reports...")

    reports = {}

    # 1. Complete cleaned dataset
    reports["cleaned_data.csv"] = df.copy()

    # 2. Summary statistics for numeric columns
    numeric_df = df.select_dtypes(include="number")

    reports["summary_statistics.csv"] = (
        numeric_df.describe()
        .T
        .reset_index()
        .rename(columns={"index": "column"})
        .round(2)
    )

    # 3. State-wise report
    state_report = (
        df.groupby("state", observed=True)
        .agg(
            households=("hhid", "count"),
            avg_monthly_expenditure=("monthly_expenditure", "mean"),
            avg_monthly_elec_bill=("avg_monthly_elec_bill", "mean"),
            house_ownership_rate=("owns_house", "mean")
        )
        .reset_index()
    )

    state_report["house_ownership_rate"] *= 100

    reports["state_wise_report.csv"] = (
        state_report.sort_values(
            "avg_monthly_elec_bill",
            ascending=False
        ).round(2)
    )

    # 4. Districts with the highest average electricity bills
    district_report = (
        df.groupby(["state", "district"], observed=True)
        .agg(
            households=("hhid", "count"),
            avg_monthly_elec_bill=("avg_monthly_elec_bill", "mean")
        )
        .reset_index()
        .sort_values(
            "avg_monthly_elec_bill",
            ascending=False
        )
    )

    reports["top10_districts_by_elec_bill.csv"] = (
        district_report.head(10).reset_index(drop=True).round(2)
    )

    # 5. Districts with the lowest average electricity bills
    reports["bottom10_districts_by_elec_bill.csv"] = (
        district_report.sort_values(
            "avg_monthly_elec_bill",
            ascending=True
        ).head(10).reset_index(drop=True).round(2)
    )

    # 6. Income category report
    reports["income_category_report.csv"] = (
        df.groupby("income_category", dropna=False, observed=True)
        .agg(
            households=("hhid", "count"),
            avg_monthly_expenditure=("monthly_expenditure", "mean"),
            avg_monthly_elec_bill=("avg_monthly_elec_bill", "mean")
        )
        .reset_index()
        .round(2)
    )

    # 7. Household size report
    reports["household_size_report.csv"] = (
        df.groupby("household_size_group", dropna=False, observed=True)
        .agg(
            households=("hhid", "count"),
            avg_household_size=("household_size", "mean"),
            avg_monthly_elec_bill=("avg_monthly_elec_bill", "mean"),
            avg_monthly_expenditure=("monthly_expenditure", "mean")
        )
        .reset_index()
        .round(2)
    )

    # 8. Monthly interview report
    monthly_df = df.copy()

    monthly_df["interview_month"] = (
        pd.to_datetime(monthly_df["interview_date"])
        .dt.to_period("M")
        .astype("string")
    )

    reports["monthly_interview_report.csv"] = (
        monthly_df.groupby("interview_month", dropna=False)
        .agg(
            interviews=("hhid", "count"),
            avg_monthly_elec_bill=("avg_monthly_elec_bill", "mean")
        )
        .reset_index()
        .round(2)
    )

    # 9. Remaining missing values report
    missing_report = pd.DataFrame({
        "column": df.columns,
        "missing_count": df.isna().sum().values,
        "missing_percentage": (
            df.isna().mean().values * 100
        ).round(2)
    })

    reports["remaining_missing_values_report.csv"] = (
        missing_report[
            missing_report["missing_count"] > 0
        ]
        .sort_values("missing_count", ascending=False)
        .reset_index(drop=True)
    )

    log.step(
        f"[Member 5] Generated {len(reports)} reports"
    )

    return reports


def save_reports(reports, df):
    """Save reports, final dataset and pipeline log."""

    # Ensure output directories exist
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Save all report CSV files
    for filename, report_df in reports.items():
        file_path = config.REPORTS_DIR / filename
        report_df.to_csv(file_path, index=False)
        log.step(f"[Member 5] Saved report: {file_path}")

    # Save the analysis-ready dataset in compressed CSV format
    final_file = config.OUTPUT_DIR / "analysis_ready_data.csv.gz"

    df.to_csv(
        final_file,
        index=False,
        compression="gzip"
    )

    log.step(
        f"[Member 5] Saved analysis-ready dataset: {final_file}"
    )

    # Save pipeline execution log
    log_file = config.OUTPUT_DIR / "pipeline_log.txt"

    log_file.write_text(
        "\n".join(log.RUN_LOG) + "\n",
        encoding="utf-8"
    )

    print(f"[Member 5] Pipeline log saved: {log_file}")
    print("[Member 5] All reports and output files saved successfully.")
