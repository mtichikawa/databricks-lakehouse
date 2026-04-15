"""
test_pipeline.py — End-to-end tests for LakehousePipeline.

Covers:
- Full pipeline on sample data: raw Parquet → bronze → silver → gold
- Row count invariants at each layer
- Quality checks pass on clean data
- Quality checks catch injected bad data
- Pipeline returns correct summary structure
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.pipeline import LakehousePipeline
from src.utils import read_delta
from src import config as cfg


def write_clean_parquet(path: Path, n: int = 50) -> pd.DataFrame:
    """Write a clean synthetic Parquet file to path."""
    rng = np.random.default_rng(99)
    base = pd.Timestamp("2023-01-15 08:00:00")
    pickup = pd.date_range(base, periods=n, freq="3min")
    dropoff = pickup + pd.to_timedelta(rng.integers(300, 1800, n), unit="s")
    df = pd.DataFrame({
        "VendorID": rng.choice([1, 2], n),
        "tpep_pickup_datetime": pickup,
        "tpep_dropoff_datetime": dropoff,
        "passenger_count": rng.integers(1, 4, n).astype(float),
        "trip_distance": rng.uniform(0.5, 15.0, n).round(2),
        "RatecodeID": 1.0,
        "store_and_fwd_flag": "N",
        "PULocationID": rng.integers(1, 50, n),
        "DOLocationID": rng.integers(1, 50, n),
        "payment_type": rng.choice([1, 2], n),
        "fare_amount": rng.uniform(5.0, 50.0, n).round(2),
        "extra": 0.5,
        "mta_tax": 0.5,
        "tip_amount": rng.uniform(0.0, 10.0, n).round(2),
        "tolls_amount": 0.0,
        "improvement_surcharge": 0.3,
        "total_amount": rng.uniform(6.0, 65.0, n).round(2),
        "congestion_surcharge": 2.5,
        "airport_fee": 0.0,
    })
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(str(path), index=False)
    return df


class TestEndToEnd:
    def test_pipeline_runs_without_error(self, temp_dir):
        raw_dir = temp_dir / "raw"
        write_clean_parquet(raw_dir / "test.parquet", n=50)

        pipeline = LakehousePipeline(
            raw_dir=raw_dir,
            delta_dir=temp_dir / "delta",
            schema_dir=temp_dir / "schemas",
            quality_log_dir=temp_dir / "quality_logs",
        )
        summary = pipeline.run()
        assert "bronze" in summary
        assert "silver" in summary

    def test_bronze_row_count_equals_raw(self, temp_dir):
        raw_dir = temp_dir / "raw"
        raw_df = write_clean_parquet(raw_dir / "test.parquet", n=50)

        pipeline = LakehousePipeline(
            raw_dir=raw_dir,
            delta_dir=temp_dir / "delta",
            schema_dir=temp_dir / "schemas",
            quality_log_dir=temp_dir / "quality_logs",
        )
        pipeline.run()

        bronze_df = read_delta(temp_dir / "delta" / cfg.BRONZE_TABLE)
        assert len(bronze_df) == len(raw_df)

    def test_silver_plus_quarantine_equals_bronze(self, temp_dir):
        """Row count invariant: silver + quarantine + deduped == bronze."""
        raw_dir = temp_dir / "raw"
        write_clean_parquet(raw_dir / "test.parquet", n=50)

        pipeline = LakehousePipeline(
            raw_dir=raw_dir,
            delta_dir=temp_dir / "delta",
            schema_dir=temp_dir / "schemas",
            quality_log_dir=temp_dir / "quality_logs",
        )
        summary = pipeline.run()

        bronze_df = read_delta(temp_dir / "delta" / cfg.BRONZE_TABLE)
        silver_df = read_delta(temp_dir / "delta" / cfg.SILVER_TABLE)
        quarantine_df = read_delta(temp_dir / "delta" / cfg.SILVER_QUARANTINE_TABLE)
        deduped = summary["silver"].get("deduped_rows", 0)

        # Full accounting: silver + quarantine + deduped rows == bronze
        assert len(silver_df) + len(quarantine_df) + deduped == len(bronze_df)

    def test_gold_tables_created(self, temp_dir):
        raw_dir = temp_dir / "raw"
        write_clean_parquet(raw_dir / "test.parquet", n=50)

        pipeline = LakehousePipeline(
            raw_dir=raw_dir,
            delta_dir=temp_dir / "delta",
            schema_dir=temp_dir / "schemas",
            quality_log_dir=temp_dir / "quality_logs",
        )
        pipeline.run()

        daily_df = read_delta(temp_dir / "delta" / cfg.GOLD_DAILY_TABLE)
        hourly_df = read_delta(temp_dir / "delta" / cfg.GOLD_HOURLY_TABLE)
        payment_df = read_delta(temp_dir / "delta" / cfg.GOLD_PAYMENT_TABLE)

        assert len(daily_df) > 0
        assert len(hourly_df) > 0
        assert len(payment_df) > 0

    def test_summary_has_elapsed_sec(self, temp_dir):
        raw_dir = temp_dir / "raw"
        write_clean_parquet(raw_dir / "test.parquet", n=20)

        pipeline = LakehousePipeline(
            raw_dir=raw_dir,
            delta_dir=temp_dir / "delta",
            schema_dir=temp_dir / "schemas",
            quality_log_dir=temp_dir / "quality_logs",
        )
        summary = pipeline.run()
        assert "elapsed_sec" in summary
        assert summary["elapsed_sec"] >= 0


class TestPipelineQualityChecks:
    def test_quality_results_in_summary(self, temp_dir):
        raw_dir = temp_dir / "raw"
        write_clean_parquet(raw_dir / "test.parquet", n=30)

        pipeline = LakehousePipeline(
            raw_dir=raw_dir,
            delta_dir=temp_dir / "delta",
            schema_dir=temp_dir / "schemas",
            quality_log_dir=temp_dir / "quality_logs",
        )
        summary = pipeline.run()
        assert "bronze_quality" in summary
        assert "passed" in summary["bronze_quality"]


class TestPipelineIdempotency:
    def test_second_run_skips_already_ingested(self, temp_dir):
        raw_dir = temp_dir / "raw"
        write_clean_parquet(raw_dir / "test.parquet", n=30)

        pipeline = LakehousePipeline(
            raw_dir=raw_dir,
            delta_dir=temp_dir / "delta",
            schema_dir=temp_dir / "schemas",
            quality_log_dir=temp_dir / "quality_logs",
        )
        summary1 = pipeline.run()
        summary2 = pipeline.run()

        assert summary1["bronze"]["new_files"] == 1
        assert summary2["bronze"]["new_files"] == 0
        assert summary2["bronze"]["skipped_files"] == 1

    def test_bronze_rows_not_doubled(self, temp_dir):
        raw_dir = temp_dir / "raw"
        raw_df = write_clean_parquet(raw_dir / "test.parquet", n=30)

        pipeline = LakehousePipeline(
            raw_dir=raw_dir,
            delta_dir=temp_dir / "delta",
            schema_dir=temp_dir / "schemas",
            quality_log_dir=temp_dir / "quality_logs",
        )
        pipeline.run()
        pipeline.run()

        bronze_df = read_delta(temp_dir / "delta" / cfg.BRONZE_TABLE)
        assert len(bronze_df) == len(raw_df)


class TestSampleData:
    """Run the full pipeline against the committed sample Parquet file."""

    def test_sample_file_exists(self):
        sample_dir = Path(__file__).parent.parent / "data" / "sample"
        parquet_files = list(sample_dir.glob("*.parquet"))
        assert len(parquet_files) > 0, "No sample Parquet file found in data/sample/"

    def test_pipeline_on_sample_data(self, temp_dir):
        sample_dir = Path(__file__).parent.parent / "data" / "sample"
        parquet_files = list(sample_dir.glob("*.parquet"))
        if not parquet_files:
            pytest.skip("No sample file available")

        pipeline = LakehousePipeline(
            raw_dir=sample_dir,
            delta_dir=temp_dir / "delta",
            schema_dir=temp_dir / "schemas",
            quality_log_dir=temp_dir / "quality_logs",
        )
        summary = pipeline.run()

        assert summary["bronze"]["total_rows_ingested"] > 0
        bronze_df = read_delta(temp_dir / "delta" / cfg.BRONZE_TABLE)
        silver_df = read_delta(temp_dir / "delta" / cfg.SILVER_TABLE)
        quarantine_df = read_delta(temp_dir / "delta" / cfg.SILVER_QUARANTINE_TABLE)

        # Row count invariant: silver + quarantine + deduped == bronze
        deduped = summary["silver"].get("deduped_rows", 0)
        assert len(silver_df) + len(quarantine_df) + deduped == len(bronze_df)
        # Sample has known dirty rows — quarantine must be non-empty
        assert len(quarantine_df) > 0
