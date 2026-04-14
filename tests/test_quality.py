"""
test_quality.py — Tests for the quality framework.

Covers:
- null_rate_check: pass/warn/fail scenarios
- row_count_match: matching/mismatched counts
- range_check: in-bounds/out-of-bounds
- uniqueness_check: unique keys/duplicate keys
- QualityRunner: runs multiple checks, returns QualitySummary
"""

import numpy as np
import pandas as pd
import pytest

from src.quality import (
    QualityCheck,
    QualityRunner,
    QualitySummary,
    freshness_check,
    null_rate_check,
    range_check,
    row_count_match,
    uniqueness_check,
)


def make_df(**kwargs) -> pd.DataFrame:
    return pd.DataFrame(kwargs)


class TestNullRateCheck:
    def test_pass_below_warn_threshold(self):
        df = make_df(fare=[1.0, 2.0, 3.0, 4.0, 5.0])
        check = null_rate_check("fare", warn_threshold=0.05, fail_threshold=0.20)
        result = check.run(df)
        assert result.status == "pass"

    def test_warn_between_thresholds(self):
        # 10% null rate; warn=0.05, fail=0.20
        df = make_df(fare=[None, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        check = null_rate_check("fare", warn_threshold=0.05, fail_threshold=0.20)
        result = check.run(df)
        assert result.status == "warn"

    def test_fail_above_fail_threshold(self):
        # 25% null rate; fail=0.20
        df = make_df(fare=[None, None, None, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0])
        check = null_rate_check("fare", warn_threshold=0.05, fail_threshold=0.20)
        result = check.run(df)
        assert result.status == "fail"

    def test_missing_column_returns_fail(self):
        df = make_df(other_col=[1, 2, 3])
        check = null_rate_check("fare")
        result = check.run(df)
        assert result.status == "fail"

    def test_empty_df_returns_pass(self):
        df = pd.DataFrame({"fare": pd.Series([], dtype=float)})
        check = null_rate_check("fare")
        result = check.run(df)
        assert result.status == "pass"

    def test_metric_value_is_float(self):
        df = make_df(fare=[1.0, None, 3.0])
        check = null_rate_check("fare")
        result = check.run(df)
        assert isinstance(result.metric_value, float)


class TestRowCountMatch:
    def test_exact_match_passes(self):
        df = make_df(x=[1, 2, 3])
        check = row_count_match(3)
        result = check.run(df)
        assert result.status == "pass"

    def test_too_few_fails(self):
        df = make_df(x=[1, 2])
        check = row_count_match(3)
        result = check.run(df)
        assert result.status == "fail"

    def test_too_many_fails(self):
        df = make_df(x=[1, 2, 3, 4])
        check = row_count_match(3)
        result = check.run(df)
        assert result.status == "fail"

    def test_metric_value_is_actual_count(self):
        df = make_df(x=[1, 2])
        check = row_count_match(3)
        result = check.run(df)
        assert result.metric_value == 2


class TestRangeCheck:
    def test_all_in_bounds_passes(self):
        df = make_df(fare=[5.0, 10.0, 15.0, 20.0])
        check = range_check("fare", min_val=1.0, max_val=500.0)
        result = check.run(df)
        assert result.status == "pass"

    def test_below_min_fails(self):
        df = make_df(fare=[0.0, 5.0, 10.0])
        check = range_check("fare", min_val=1.0, max_val=500.0)
        result = check.run(df)
        assert result.status == "fail"

    def test_above_max_fails(self):
        df = make_df(fare=[5.0, 10.0, 600.0])
        check = range_check("fare", min_val=1.0, max_val=500.0)
        result = check.run(df)
        assert result.status == "fail"

    def test_missing_column_fails(self):
        df = make_df(other=[1.0])
        check = range_check("fare", min_val=0.0, max_val=100.0)
        result = check.run(df)
        assert result.status == "fail"

    def test_metric_value_is_violation_count(self):
        df = make_df(fare=[0.0, 0.0, 5.0])
        check = range_check("fare", min_val=1.0, max_val=100.0)
        result = check.run(df)
        assert result.metric_value == 2


class TestUniquenessCheck:
    def test_unique_keys_passes(self):
        df = make_df(date=["2023-01-01", "2023-01-02", "2023-01-03"], zone=[1, 1, 1])
        check = uniqueness_check(["date", "zone"])
        result = check.run(df)
        assert result.status == "pass"

    def test_duplicate_keys_fails(self):
        df = make_df(date=["2023-01-01", "2023-01-01"], zone=[1, 1])
        check = uniqueness_check(["date", "zone"])
        result = check.run(df)
        assert result.status == "fail"

    def test_duplicate_count_in_metric(self):
        df = make_df(date=["A", "A", "A", "B"], zone=[1, 1, 1, 2])
        check = uniqueness_check(["date", "zone"])
        result = check.run(df)
        # 3 A/1 rows → 2 duplicates
        assert result.metric_value == 2

    def test_missing_column_fails(self):
        df = make_df(date=["2023-01-01"])
        check = uniqueness_check(["date", "zone"])
        result = check.run(df)
        assert result.status == "fail"


class TestFreshnessCheck:
    def test_has_data_passes(self):
        df = make_df(pickup_at=[pd.Timestamp("2023-01-15")])
        check = freshness_check("pickup_at", min_rows=1)
        result = check.run(df)
        assert result.status == "pass"

    def test_empty_fails(self):
        df = make_df(pickup_at=pd.Series([], dtype="datetime64[ns]"))
        check = freshness_check("pickup_at", min_rows=1)
        result = check.run(df)
        assert result.status == "fail"


class TestQualityRunner:
    def test_all_pass_summary_passed(self):
        df = make_df(x=[1, 2, 3])
        checks = [row_count_match(3), uniqueness_check(["x"])]
        runner = QualityRunner(checks, source_table="src", target_table="tgt")
        summary = runner.run_all(df)
        assert summary.passed is True
        assert summary.has_failures is False

    def test_one_fail_summary_not_passed(self):
        df = make_df(x=[1, 2])
        checks = [row_count_match(3)]
        runner = QualityRunner(checks)
        summary = runner.run_all(df)
        assert summary.passed is False
        assert summary.has_failures is True

    def test_summary_has_results_list(self):
        df = make_df(x=[1])
        checks = [row_count_match(1), uniqueness_check(["x"])]
        runner = QualityRunner(checks)
        summary = runner.run_all(df)
        assert len(summary.results) == 2

    def test_log_results_writes_file(self, temp_dir):
        df = make_df(x=[1, 2, 3])
        checks = [row_count_match(3)]
        runner = QualityRunner(checks, source_table="a", target_table="b")
        summary = runner.run_all(df)
        path = runner.log_results(summary, log_dir=temp_dir)
        assert path.exists()
        assert path.suffix == ".json"

    def test_check_exception_returns_fail(self):
        def bad_fn(df):
            raise ValueError("boom")

        check = QualityCheck("exploding", bad_fn)
        result = check.run(pd.DataFrame())
        assert result.status == "fail"
        assert "boom" in result.message
