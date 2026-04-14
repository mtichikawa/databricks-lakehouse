"""
test_silver.py — Tests for SilverTransformer.

Covers:
- Column renaming (snake_case)
- Type casting
- Derived columns (trip_duration_min, fare_per_mile, fare_per_minute)
- Quarantine logic (null timestamp, zero fare, extreme distance, zero passengers)
- Deduplication
- Valid rows pass through unchanged
- Quarantine preserves _quarantine_reason column
"""

import pandas as pd
import pytest

from src.silver import SilverTransformer


class TestColumnRenaming:
    def test_vendor_id_renamed(self, sample_bronze_df):
        t = SilverTransformer()
        clean, _ = t.clean(sample_bronze_df)
        assert "vendor_id" in clean.columns
        assert "VendorID" not in clean.columns

    def test_pickup_at_renamed(self, sample_bronze_df):
        t = SilverTransformer()
        clean, _ = t.clean(sample_bronze_df)
        assert "pickup_at" in clean.columns
        assert "tpep_pickup_datetime" not in clean.columns

    def test_dropoff_at_renamed(self, sample_bronze_df):
        t = SilverTransformer()
        clean, _ = t.clean(sample_bronze_df)
        assert "dropoff_at" in clean.columns

    def test_pickup_location_id_renamed(self, sample_bronze_df):
        t = SilverTransformer()
        clean, _ = t.clean(sample_bronze_df)
        assert "pickup_location_id" in clean.columns
        assert "PULocationID" not in clean.columns

    def test_fare_amount_retained(self, sample_bronze_df):
        t = SilverTransformer()
        clean, _ = t.clean(sample_bronze_df)
        assert "fare_amount" in clean.columns


class TestTypeCasting:
    def test_pickup_at_is_datetime(self, sample_bronze_df):
        t = SilverTransformer()
        clean, _ = t.clean(sample_bronze_df)
        assert pd.api.types.is_datetime64_any_dtype(clean["pickup_at"])

    def test_fare_amount_is_float(self, sample_bronze_df):
        t = SilverTransformer()
        clean, _ = t.clean(sample_bronze_df)
        assert clean["fare_amount"].dtype in (float, "float64")


class TestDerivedColumns:
    def test_trip_duration_min_present(self, sample_bronze_df):
        t = SilverTransformer()
        clean, _ = t.clean(sample_bronze_df)
        assert "trip_duration_min" in clean.columns

    def test_trip_duration_min_positive(self, sample_bronze_df):
        t = SilverTransformer()
        clean, _ = t.clean(sample_bronze_df)
        # All clean rows should have positive duration
        assert (clean["trip_duration_min"] > 0).all()

    def test_fare_per_mile_present(self, sample_bronze_df):
        t = SilverTransformer()
        clean, _ = t.clean(sample_bronze_df)
        assert "fare_per_mile" in clean.columns

    def test_fare_per_minute_present(self, sample_bronze_df):
        t = SilverTransformer()
        clean, _ = t.clean(sample_bronze_df)
        assert "fare_per_minute" in clean.columns

    def test_duration_calculation_known_input(self):
        """Known input → expected duration."""
        df = pd.DataFrame({
            "VendorID": [1],
            "tpep_pickup_datetime": [pd.Timestamp("2023-01-15 10:00:00")],
            "tpep_dropoff_datetime": [pd.Timestamp("2023-01-15 10:30:00")],
            "passenger_count": [2.0],
            "trip_distance": [5.0],
            "RatecodeID": [1.0],
            "store_and_fwd_flag": ["N"],
            "PULocationID": [100],
            "DOLocationID": [200],
            "payment_type": [1],
            "fare_amount": [15.0],
            "extra": [0.5],
            "mta_tax": [0.5],
            "tip_amount": [3.0],
            "tolls_amount": [0.0],
            "improvement_surcharge": [0.3],
            "total_amount": [19.3],
            "congestion_surcharge": [2.5],
            "airport_fee": [0.0],
            "_source_file": ["test.parquet"],
            "_ingested_at": [pd.Timestamp("2023-02-01")],
            "_batch_id": ["batch-1"],
        })
        t = SilverTransformer()
        clean, _ = t.clean(df)
        assert abs(clean["trip_duration_min"].iloc[0] - 30.0) < 0.01

    def test_fare_per_mile_calculation(self):
        """fare_per_mile = fare / distance."""
        df = pd.DataFrame({
            "VendorID": [1],
            "tpep_pickup_datetime": [pd.Timestamp("2023-01-15 10:00:00")],
            "tpep_dropoff_datetime": [pd.Timestamp("2023-01-15 10:30:00")],
            "passenger_count": [1.0],
            "trip_distance": [4.0],
            "RatecodeID": [1.0],
            "store_and_fwd_flag": ["N"],
            "PULocationID": [1],
            "DOLocationID": [2],
            "payment_type": [1],
            "fare_amount": [20.0],
            "extra": [0.0],
            "mta_tax": [0.5],
            "tip_amount": [4.0],
            "tolls_amount": [0.0],
            "improvement_surcharge": [0.3],
            "total_amount": [24.8],
            "congestion_surcharge": [0.0],
            "airport_fee": [0.0],
            "_source_file": ["test.parquet"],
            "_ingested_at": [pd.Timestamp("2023-02-01")],
            "_batch_id": ["b"],
        })
        t = SilverTransformer()
        clean, _ = t.clean(df)
        assert abs(clean["fare_per_mile"].iloc[0] - 5.0) < 0.01

    def test_payment_type_label_added(self, sample_bronze_df):
        t = SilverTransformer()
        clean, _ = t.clean(sample_bronze_df)
        assert "payment_type" in clean.columns


class TestQuarantine:
    """Invalid rows go to quarantine, not silently dropped."""

    def _make_base_row(self):
        return {
            "VendorID": 1,
            "tpep_pickup_datetime": pd.Timestamp("2023-01-15 10:00:00"),
            "tpep_dropoff_datetime": pd.Timestamp("2023-01-15 10:30:00"),
            "passenger_count": 2.0,
            "trip_distance": 5.0,
            "RatecodeID": 1.0,
            "store_and_fwd_flag": "N",
            "PULocationID": 100,
            "DOLocationID": 200,
            "payment_type": 1,
            "fare_amount": 15.0,
            "extra": 0.5,
            "mta_tax": 0.5,
            "tip_amount": 3.0,
            "tolls_amount": 0.0,
            "improvement_surcharge": 0.3,
            "total_amount": 19.3,
            "congestion_surcharge": 2.5,
            "airport_fee": 0.0,
            "_source_file": "test.parquet",
            "_ingested_at": pd.Timestamp("2023-02-01"),
            "_batch_id": "b",
        }

    def test_null_pickup_quarantined(self):
        row = self._make_base_row()
        row["tpep_pickup_datetime"] = None
        df = pd.DataFrame([row])
        t = SilverTransformer()
        clean, q = t.clean(df)
        assert len(clean) == 0
        assert len(q) == 1
        assert "_quarantine_reason" in q.columns

    def test_zero_fare_quarantined(self):
        row = self._make_base_row()
        row["fare_amount"] = 0.0
        df = pd.DataFrame([row])
        t = SilverTransformer()
        clean, q = t.clean(df)
        assert len(q) == 1

    def test_negative_fare_quarantined(self):
        row = self._make_base_row()
        row["fare_amount"] = -5.0
        df = pd.DataFrame([row])
        t = SilverTransformer()
        clean, q = t.clean(df)
        assert len(q) == 1

    def test_extreme_distance_quarantined(self):
        row = self._make_base_row()
        row["trip_distance"] = 300.0  # > 200 miles
        df = pd.DataFrame([row])
        t = SilverTransformer()
        clean, q = t.clean(df)
        assert len(q) == 1

    def test_zero_passengers_quarantined(self):
        row = self._make_base_row()
        row["passenger_count"] = 0.0
        df = pd.DataFrame([row])
        t = SilverTransformer()
        clean, q = t.clean(df)
        assert len(q) == 1

    def test_quarantine_reason_column_present(self):
        row = self._make_base_row()
        row["fare_amount"] = 0.0
        df = pd.DataFrame([row])
        t = SilverTransformer()
        _, q = t.clean(df)
        assert "_quarantine_reason" in q.columns
        assert q["_quarantine_reason"].iloc[0] != ""

    def test_valid_row_not_quarantined(self, sample_bronze_df):
        t = SilverTransformer()
        clean, q = t.clean(sample_bronze_df)
        assert len(clean) == len(sample_bronze_df)
        assert len(q) == 0


class TestDeduplication:
    def test_duplicate_rows_collapsed(self, sample_bronze_df):
        # Append an exact copy
        doubled = pd.concat([sample_bronze_df, sample_bronze_df], ignore_index=True)
        t = SilverTransformer()
        clean, _ = t.clean(doubled)
        # Should end up with original row count (duplicates removed)
        assert len(clean) == len(sample_bronze_df)

    def test_unique_rows_not_removed(self, sample_bronze_df):
        t = SilverTransformer()
        clean, _ = t.clean(sample_bronze_df)
        assert len(clean) == len(sample_bronze_df)
