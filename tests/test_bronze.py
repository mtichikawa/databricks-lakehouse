"""
test_bronze.py — Tests for BronzeIngester.

Covers:
- Metadata columns added correctly
- Idempotency: ingesting same file twice doesn't duplicate rows
- Schema preservation: output columns include input + metadata
- Empty directory handling
"""

from pathlib import Path

import pandas as pd
import pytest

from src.bronze import BronzeIngester
from src import config as cfg


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def write_sample_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(str(path), index=False)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestBronzeMetadata:
    """Metadata columns are added correctly on ingest."""

    def test_metadata_columns_added(self, temp_dir, sample_raw_df):
        raw_dir = temp_dir / "raw"
        delta_dir = temp_dir / "delta"
        parquet_path = raw_dir / "yellow_tripdata_2023-01.parquet"
        write_sample_parquet(sample_raw_df, parquet_path)

        ingester = BronzeIngester(raw_dir=raw_dir, delta_dir=delta_dir)
        ingester.ingest_file(parquet_path)

        from src.utils import read_delta
        bronze_df = read_delta(delta_dir / cfg.BRONZE_TABLE)

        assert "_source_file" in bronze_df.columns
        assert "_ingested_at" in bronze_df.columns
        assert "_batch_id" in bronze_df.columns

    def test_source_file_value(self, temp_dir, sample_raw_df):
        raw_dir = temp_dir / "raw"
        delta_dir = temp_dir / "delta"
        parquet_path = raw_dir / "yellow_tripdata_2023-01.parquet"
        write_sample_parquet(sample_raw_df, parquet_path)

        ingester = BronzeIngester(raw_dir=raw_dir, delta_dir=delta_dir)
        ingester.ingest_file(parquet_path)

        from src.utils import read_delta
        bronze_df = read_delta(delta_dir / cfg.BRONZE_TABLE)

        assert bronze_df["_source_file"].unique().tolist() == ["yellow_tripdata_2023-01.parquet"]

    def test_ingested_at_is_datetime(self, temp_dir, sample_raw_df):
        raw_dir = temp_dir / "raw"
        delta_dir = temp_dir / "delta"
        parquet_path = raw_dir / "yellow_tripdata_2023-01.parquet"
        write_sample_parquet(sample_raw_df, parquet_path)

        ingester = BronzeIngester(raw_dir=raw_dir, delta_dir=delta_dir)
        ingester.ingest_file(parquet_path)

        from src.utils import read_delta
        bronze_df = read_delta(delta_dir / cfg.BRONZE_TABLE)

        # Should be parseable as datetime
        dt = pd.to_datetime(bronze_df["_ingested_at"])
        assert dt.notna().all()

    def test_batch_id_is_string(self, temp_dir, sample_raw_df):
        raw_dir = temp_dir / "raw"
        delta_dir = temp_dir / "delta"
        parquet_path = raw_dir / "test.parquet"
        write_sample_parquet(sample_raw_df, parquet_path)

        ingester = BronzeIngester(raw_dir=raw_dir, delta_dir=delta_dir)
        ingester.ingest_file(parquet_path, batch_id="my-batch-42")

        from src.utils import read_delta
        bronze_df = read_delta(delta_dir / cfg.BRONZE_TABLE)

        assert (bronze_df["_batch_id"] == "my-batch-42").all()


class TestBronzeSchemaPreservation:
    """Output schema includes original columns plus metadata."""

    def test_original_columns_preserved(self, temp_dir, sample_raw_df):
        raw_dir = temp_dir / "raw"
        delta_dir = temp_dir / "delta"
        parquet_path = raw_dir / "test.parquet"
        write_sample_parquet(sample_raw_df, parquet_path)

        ingester = BronzeIngester(raw_dir=raw_dir, delta_dir=delta_dir)
        ingester.ingest_file(parquet_path)

        from src.utils import read_delta
        bronze_df = read_delta(delta_dir / cfg.BRONZE_TABLE)

        for col in sample_raw_df.columns:
            assert col in bronze_df.columns, f"Missing column: {col}"

    def test_row_count_matches(self, temp_dir, sample_raw_df):
        raw_dir = temp_dir / "raw"
        delta_dir = temp_dir / "delta"
        parquet_path = raw_dir / "test.parquet"
        write_sample_parquet(sample_raw_df, parquet_path)

        ingester = BronzeIngester(raw_dir=raw_dir, delta_dir=delta_dir)
        rows = ingester.ingest_file(parquet_path)

        assert rows == len(sample_raw_df)

        from src.utils import read_delta
        bronze_df = read_delta(delta_dir / cfg.BRONZE_TABLE)
        assert len(bronze_df) == len(sample_raw_df)


class TestBronzeIdempotency:
    """Ingesting the same file twice must not duplicate rows."""

    def test_ingest_all_skips_already_ingested(self, temp_dir, sample_raw_df):
        raw_dir = temp_dir / "raw"
        delta_dir = temp_dir / "delta"
        parquet_path = raw_dir / "yellow_tripdata_2023-01.parquet"
        write_sample_parquet(sample_raw_df, parquet_path)

        ingester = BronzeIngester(raw_dir=raw_dir, delta_dir=delta_dir)

        summary1 = ingester.ingest_all()
        summary2 = ingester.ingest_all()

        assert summary1["new_files"] == 1
        assert summary1["skipped_files"] == 0
        assert summary2["new_files"] == 0
        assert summary2["skipped_files"] == 1

        from src.utils import read_delta
        bronze_df = read_delta(delta_dir / cfg.BRONZE_TABLE)
        # Rows must not be doubled
        assert len(bronze_df) == len(sample_raw_df)

    def test_get_ingested_files_returns_set(self, temp_dir, sample_raw_df):
        raw_dir = temp_dir / "raw"
        delta_dir = temp_dir / "delta"
        parquet_path = raw_dir / "test.parquet"
        write_sample_parquet(sample_raw_df, parquet_path)

        ingester = BronzeIngester(raw_dir=raw_dir, delta_dir=delta_dir)

        # Before ingest — empty set
        assert ingester.get_ingested_files() == set()

        ingester.ingest_file(parquet_path)

        # After ingest — contains the filename
        assert "test.parquet" in ingester.get_ingested_files()


class TestBronzeEmptyDirectory:
    """Handles missing / empty raw directory gracefully."""

    def test_empty_raw_dir_returns_zero_summary(self, temp_dir):
        raw_dir = temp_dir / "raw_empty"
        delta_dir = temp_dir / "delta"

        ingester = BronzeIngester(raw_dir=raw_dir, delta_dir=delta_dir)
        summary = ingester.ingest_all()

        assert summary["total_files"] == 0
        assert summary["new_files"] == 0

    def test_list_raw_files_nonexistent_dir(self, temp_dir):
        ingester = BronzeIngester(
            raw_dir=temp_dir / "does_not_exist",
            delta_dir=temp_dir / "delta",
        )
        assert ingester.list_raw_files() == []
