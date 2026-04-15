"""
test_gold.py — Tests for GoldAggregator.

Covers:
- daily_stats aggregation: known 5-row input → expected group counts and sums
- hourly_demand: rows at different hours → correct hour buckets
- payment_mix: pct_of_daily_trips sums to 1.0 per date
- No duplicate keys in output tables
"""

import pandas as pd
import pytest

from src.gold import GoldAggregator


def make_silver_df(rows: list[dict]) -> pd.DataFrame:
    """Helper to build a silver-schema DataFrame from a list of dicts."""
    defaults = {
        "vendor_id": 1,
        "pickup_at": pd.Timestamp("2023-01-15 10:00:00"),
        "dropoff_at": pd.Timestamp("2023-01-15 10:30:00"),
        "passenger_count": 2.0,
        "trip_distance": 5.0,
        "pickup_location_id": 100,
        "dropoff_location_id": 200,
        "payment_type": "credit_card",
        "payment_type_code": 1,
        "fare_amount": 15.0,
        "tip_amount": 3.0,
        "total_amount": 18.0,
        "trip_duration_min": 30.0,
        "fare_per_mile": 3.0,
        "fare_per_minute": 0.5,
    }
    return pd.DataFrame([{**defaults, **r} for r in rows])


class TestDailyStats:
    def test_aggregation_groups_correctly(self):
        """5 rows split across 2 zones → 2 groups."""
        rows = [
            {"pickup_location_id": 1, "fare_amount": 10.0, "total_amount": 12.0},
            {"pickup_location_id": 1, "fare_amount": 20.0, "total_amount": 22.0},
            {"pickup_location_id": 2, "fare_amount": 30.0, "total_amount": 33.0},
            {"pickup_location_id": 2, "fare_amount": 40.0, "total_amount": 44.0},
            {"pickup_location_id": 2, "fare_amount": 50.0, "total_amount": 55.0},
        ]
        df = make_silver_df(rows)
        agg = GoldAggregator()
        result = agg.build_daily_stats(df)

        zone1 = result[result["pickup_zone"] == 1]
        zone2 = result[result["pickup_zone"] == 2]
        assert zone1["trip_count"].iloc[0] == 2
        assert zone2["trip_count"].iloc[0] == 3

    def test_total_revenue_sum(self):
        rows = [
            {"pickup_location_id": 1, "total_amount": 10.0},
            {"pickup_location_id": 1, "total_amount": 20.0},
        ]
        df = make_silver_df(rows)
        result = GoldAggregator().build_daily_stats(df)
        zone1 = result[result["pickup_zone"] == 1]
        assert abs(zone1["total_revenue"].iloc[0] - 30.0) < 0.01

    def test_avg_fare_calculation(self):
        rows = [
            {"pickup_location_id": 5, "fare_amount": 10.0},
            {"pickup_location_id": 5, "fare_amount": 30.0},
        ]
        df = make_silver_df(rows)
        result = GoldAggregator().build_daily_stats(df)
        zone5 = result[result["pickup_zone"] == 5]
        assert abs(zone5["avg_fare"].iloc[0] - 20.0) < 0.01

    def test_no_duplicate_date_zone_keys(self, sample_silver_df):
        result = GoldAggregator().build_daily_stats(sample_silver_df)
        dupes = result.duplicated(subset=["trip_date", "pickup_zone"]).sum()
        assert dupes == 0

    def test_empty_df_returns_empty(self):
        result = GoldAggregator().build_daily_stats(pd.DataFrame())
        assert len(result) == 0

    def test_expected_columns_present(self, sample_silver_df):
        result = GoldAggregator().build_daily_stats(sample_silver_df)
        for col in ["trip_date", "pickup_zone", "trip_count", "total_revenue", "avg_fare"]:
            assert col in result.columns


class TestHourlyDemand:
    def test_hour_buckets_correct(self):
        """Trips at different hours → correct hour column values."""
        rows = [
            {"pickup_at": pd.Timestamp("2023-01-15 08:30:00"), "pickup_location_id": 1},
            {"pickup_at": pd.Timestamp("2023-01-15 14:15:00"), "pickup_location_id": 1},
            {"pickup_at": pd.Timestamp("2023-01-15 22:45:00"), "pickup_location_id": 1},
        ]
        df = make_silver_df(rows)
        result = GoldAggregator().build_hourly_demand(df)

        hours = sorted(result["hour"].tolist())
        assert 8 in hours
        assert 14 in hours
        assert 22 in hours

    def test_trip_count_per_hour(self):
        """3 trips at 8am, 1 at 9am → correct counts."""
        rows = [
            {"pickup_at": pd.Timestamp("2023-01-15 08:00:00")},
            {"pickup_at": pd.Timestamp("2023-01-15 08:30:00")},
            {"pickup_at": pd.Timestamp("2023-01-15 08:59:00")},
            {"pickup_at": pd.Timestamp("2023-01-15 09:00:00")},
        ]
        df = make_silver_df(rows)
        result = GoldAggregator().build_hourly_demand(df)

        hour8 = result[result["hour"] == 8]
        hour9 = result[result["hour"] == 9]
        assert hour8["trip_count"].iloc[0] == 3
        assert hour9["trip_count"].iloc[0] == 1

    def test_empty_df_returns_empty(self):
        result = GoldAggregator().build_hourly_demand(pd.DataFrame())
        assert len(result) == 0


class TestPaymentMix:
    def test_pct_sums_to_one_per_date(self):
        """pct_of_daily_trips must sum to 1.0 for each date."""
        rows = [
            {"payment_type": "credit_card", "fare_amount": 10.0, "total_amount": 12.0},
            {"payment_type": "credit_card", "fare_amount": 15.0, "total_amount": 17.0},
            {"payment_type": "cash", "fare_amount": 20.0, "total_amount": 22.0},
        ]
        df = make_silver_df(rows)
        result = GoldAggregator().build_payment_mix(df)

        for date, grp in result.groupby("trip_date"):
            total_pct = grp["pct_of_daily_trips"].sum()
            assert abs(total_pct - 1.0) < 1e-6, f"pct sum={total_pct} for date={date}"

    def test_trip_count_per_payment_type(self):
        rows = [
            {"payment_type": "credit_card"},
            {"payment_type": "credit_card"},
            {"payment_type": "cash"},
        ]
        df = make_silver_df(rows)
        result = GoldAggregator().build_payment_mix(df)

        cc = result[result["payment_type"] == "credit_card"]
        cash = result[result["payment_type"] == "cash"]
        assert cc["trip_count"].iloc[0] == 2
        assert cash["trip_count"].iloc[0] == 1

    def test_no_duplicate_date_payment_keys(self, sample_silver_df):
        result = GoldAggregator().build_payment_mix(sample_silver_df)
        dupes = result.duplicated(subset=["trip_date", "payment_type"]).sum()
        assert dupes == 0

    def test_empty_df_returns_empty(self):
        result = GoldAggregator().build_payment_mix(pd.DataFrame())
        assert len(result) == 0
