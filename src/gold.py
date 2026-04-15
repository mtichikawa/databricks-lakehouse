"""
gold.py — GoldAggregator: silver → gold aggregate tables.

Three gold tables:
- gold_daily_stats:   one row per (date, pickup_zone)
- gold_hourly_demand: one row per (date, hour, pickup_zone)
- gold_payment_mix:   one row per (date, payment_type) with pct_of_daily_trips
"""

from pathlib import Path
from typing import Optional

import pandas as pd

from . import config as cfg
from .utils import get_logger, overwrite_delta

logger = get_logger(__name__)


class GoldAggregator:
    """
    Builds business-level aggregate tables from the silver layer.

    Usage:
        agg = GoldAggregator()
        summary = agg.build_all(silver_df)
    """

    def __init__(self, delta_dir: Optional[Path] = None):
        self.delta_dir = Path(delta_dir) if delta_dir else cfg.DELTA_DIR
        self.daily_path = self.delta_dir / cfg.GOLD_DAILY_TABLE
        self.hourly_path = self.delta_dir / cfg.GOLD_HOURLY_TABLE
        self.payment_path = self.delta_dir / cfg.GOLD_PAYMENT_TABLE

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_all(self, silver_df: pd.DataFrame) -> dict:
        """
        Build all three gold tables from silver_df.

        Returns a summary dict with row counts for each table.
        """
        logger.info("Building gold tables from %d silver rows", len(silver_df))

        daily = self.build_daily_stats(silver_df)
        hourly = self.build_hourly_demand(silver_df)
        payment = self.build_payment_mix(silver_df)

        overwrite_delta(daily, self.daily_path)
        overwrite_delta(hourly, self.hourly_path)
        overwrite_delta(payment, self.payment_path)

        summary = {
            "gold_daily_stats_rows": len(daily),
            "gold_hourly_demand_rows": len(hourly),
            "gold_payment_mix_rows": len(payment),
        }
        logger.info("Gold build complete: %s", summary)
        return summary

    def build_daily_stats(self, silver_df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate silver trips to one row per (date, pickup_zone).

        Columns:
            trip_date, pickup_zone, trip_count, total_revenue,
            avg_fare, avg_tip_pct, avg_duration_min, avg_distance_miles
        """
        df = silver_df.copy()
        if df.empty:
            return pd.DataFrame(columns=[
                "trip_date", "pickup_zone", "trip_count", "total_revenue",
                "avg_fare", "avg_tip_pct", "avg_duration_min", "avg_distance_miles",
            ])

        df["trip_date"] = pd.to_datetime(df["pickup_at"]).dt.date
        df["pickup_zone"] = df.get("pickup_location_id", pd.Series(dtype=object))

        # tip percentage = tip_amount / fare_amount (where fare > 0)
        df["tip_pct"] = df["tip_amount"] / df["fare_amount"].replace(0, float("nan"))

        agg = (
            df.groupby(["trip_date", "pickup_zone"], dropna=False)
            .agg(
                trip_count=("fare_amount", "count"),
                total_revenue=("total_amount", "sum"),
                avg_fare=("fare_amount", "mean"),
                avg_tip_pct=("tip_pct", "mean"),
                avg_duration_min=("trip_duration_min", "mean"),
                avg_distance_miles=("trip_distance", "mean"),
            )
            .reset_index()
        )

        logger.debug("gold_daily_stats: %d rows", len(agg))
        return agg

    def build_hourly_demand(self, silver_df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate silver trips to one row per (date, hour, pickup_zone).

        Columns: trip_date, hour, pickup_zone, trip_count, avg_fare
        """
        df = silver_df.copy()
        if df.empty:
            return pd.DataFrame(columns=[
                "trip_date", "hour", "pickup_zone", "trip_count", "avg_fare",
            ])

        df["trip_date"] = pd.to_datetime(df["pickup_at"]).dt.date
        df["hour"] = pd.to_datetime(df["pickup_at"]).dt.hour
        df["pickup_zone"] = df.get("pickup_location_id", pd.Series(dtype=object))

        agg = (
            df.groupby(["trip_date", "hour", "pickup_zone"], dropna=False)
            .agg(
                trip_count=("fare_amount", "count"),
                avg_fare=("fare_amount", "mean"),
            )
            .reset_index()
        )

        logger.debug("gold_hourly_demand: %d rows", len(agg))
        return agg

    def build_payment_mix(self, silver_df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate silver trips to one row per (date, payment_type).

        Columns: trip_date, payment_type, trip_count, total_revenue, pct_of_daily_trips
        """
        df = silver_df.copy()
        if df.empty:
            return pd.DataFrame(columns=[
                "trip_date", "payment_type", "trip_count",
                "total_revenue", "pct_of_daily_trips",
            ])

        df["trip_date"] = pd.to_datetime(df["pickup_at"]).dt.date
        payment_col = "payment_type" if "payment_type" in df.columns else "payment_type_code"

        agg = (
            df.groupby(["trip_date", payment_col], dropna=False)
            .agg(
                trip_count=("fare_amount", "count"),
                total_revenue=("total_amount", "sum"),
            )
            .reset_index()
        )

        if payment_col != "payment_type":
            agg = agg.rename(columns={payment_col: "payment_type"})

        # Percentage of daily trips per payment type
        daily_totals = agg.groupby("trip_date")["trip_count"].transform("sum")
        agg["pct_of_daily_trips"] = agg["trip_count"] / daily_totals

        logger.debug("gold_payment_mix: %d rows", len(agg))
        return agg
