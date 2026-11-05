import pytest

import pipeline
from etl import member5_reports_load as m5
from etl import config

if not m5.IMPLEMENTED:
    pytest.skip("Member 5 not merged yet", allow_module_level=True)


@pytest.fixture(scope="module")
def clean():
    return pipeline.transform(pipeline.extract())


def test_at_least_three_reports(clean):
    reports = m5.generate_reports(clean)
    assert len(reports) >= 3
    assert "state_wise_report.csv" in reports


def test_top10_has_ten_rows(clean):
    assert len(m5.generate_reports(clean)["top10_districts_by_elec_bill.csv"]) == 10


def test_full_pipeline_writes_files(clean):
    pipeline.run_etl_pipeline()
    assert (config.REPORTS_DIR / "state_wise_report.csv").exists()
    assert (config.OUTPUT_DIR / "pipeline_log.txt").exists()
