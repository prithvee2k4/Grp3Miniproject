"""Check one member's work.  Usage:  python check_member.py 1   (or 2, 3, 4, 5)"""
import sys

import pipeline
from etl import log


def main(n):
    log.reset()
    df = pipeline.extract()
    print(f"\nCHECK OK after Member 1: {df.shape[0]:,} rows x {df.shape[1]} columns")
    if n >= 2:
        from etl import member2_structural_cleaning as m2
        df = m2.run(df)
        print(f"CHECK OK after Member 2: {df.shape[0]:,} rows x {df.shape[1]} columns")
    if n >= 3:
        from etl import member3_missing_invalid as m3
        df = m3.run(df)
        key = ["land_area_sqft", "monthly_expenditure", "avg_monthly_elec_bill"]
        print(f"CHECK OK after Member 3: {df.shape[0]:,} rows x {df.shape[1]} columns, "
              f"{int(df[key].isna().sum().sum())} missing in the 3 key money/area columns")
    if n >= 4:
        from etl import member4_features as m4
        df = m4.run(df)
        print(f"CHECK OK after Member 4: {df.shape[0]:,} rows x {df.shape[1]} columns")
    if n >= 5:
        pipeline.run_etl_pipeline()
        print(f"CHECK OK after Member 5: full pipeline finished, "
              f"{len(list(pipeline.config.REPORTS_DIR.glob('*.csv')))} report files in reports/")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 5)
