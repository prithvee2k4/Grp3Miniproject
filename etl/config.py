"""Shared settings for the CEEW IRES 2020 ETL pipeline."""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "raw_data"
REPORTS_DIR = BASE_DIR / "reports"
OUTPUT_DIR = BASE_DIR / "output"

SOURCE_FILE = RAW_DIR / "CEEW20IRES_20Data.csv"
RAW_COPY = RAW_DIR / "ceew_ires_raw_copy.csv"

# Columns that must exist in the raw file (used by the validation step)
REQUIRED_COLUMNS = ["hhid", "s_name", "d_name", "interview_date", "q202_resp_age",
                    "q213_no_members", "q234_month_exp", "avg_monthly_bill"]

# Friendly names for the columns used in the analysis (everything else keeps its code name)
RENAME_MAP = {
    "s_name": "state",
    "state_abbv": "state_abbr",
    "s_code": "state_code",
    "d_name": "district",
    "d_code": "district_code",
    "village_ward_name": "village_ward",
    "q103_survey_type": "survey_type_code",
    "q201_resp_gender": "respondent_gender_code",
    "q202_resp_age": "respondent_age",
    "q211_caste": "caste_code",
    "q213_no_members": "household_size",
    "q214_no_members_lessthan18": "members_under_18",
    "q214_no_members_18_40": "members_18_40",
    "q214_no_members_41_60": "members_41_60",
    "q214_no_members_morethan60": "members_over_60",
    "q220_own_house_yn": "owns_house",
    "q223_house_no_bedrooms": "bedrooms",
    "land_area_sqfeet": "land_area_sqft",
    "q234_month_exp": "monthly_expenditure",
    "q236_income_category": "income_category",
    "avg_monthly_bill": "avg_monthly_elec_bill",
}

# Cleaning rules
MAX_MISSING_SHARE = 0.90          # drop columns that are more than 90% empty
DONT_KNOW_CODES = (88, 99)        # survey codes meaning "don't know / refused"
VALID_AGE = (18, 100)
OUTLIER_PERCENTILES = (0.01, 0.99)

# Columns used by the imputation, validation and outlier steps
KEY_NUMERIC = ["land_area_sqft", "monthly_expenditure", "avg_monthly_elec_bill"]
KEY_CATEGORY_CODES = ["income_category", "caste_code"]
