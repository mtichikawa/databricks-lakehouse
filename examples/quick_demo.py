"""
quick_demo.py — Full lakehouse pipeline demo on bundled sample data.

No Databricks, no downloads, no API keys required.
Uses the deltalake Python library for local Delta table I/O.

Run from project root:
    python examples/quick_demo.py
"""

import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline import LakehousePipeline
from src.utils import get_logger, read_delta
from src import config as cfg

logger = get_logger("quick_demo")


def print_section(title: str) -> None:
    width = 60
    print(f"\n{'=' * width}")
    print(f"  {title}")
    print(f"{'=' * width}")


def print_table(df, max_rows: int = 5, title: str = "") -> None:
    if title:
        print(f"\n  {title}")
    if df.empty:
        print("  (empty)")
        return
    try:
        import pandas as pd
        with pd.option_context("display.max_columns", 8, "display.width", 100):
            print(df.head(max_rows).to_string(index=False))
        if len(df) > max_rows:
            print(f"  ... ({len(df)} total rows)")
    except Exception:
        print(repr(df.head(max_rows)))


def main():
    sample_dir = PROJECT_ROOT / "data" / "sample"
    parquet_files = list(sample_dir.glob("*.parquet"))

    if not parquet_files:
        print(
            "\nNo sample Parquet file found in data/sample/.\n"
            "Run: python scripts/create_sample.py\n"
        )
        sys.exit(1)

    print_section("Databricks Medallion Lakehouse — Local Demo")
    print(f"  Sample file: {parquet_files[0].name}")
    print(f"  Using deltalake Python library (no Spark needed)\n")

    with tempfile.TemporaryDirectory(prefix="lakehouse_demo_") as tmp:
        tmp_path = Path(tmp)

        pipeline = LakehousePipeline(
            raw_dir=sample_dir,
            delta_dir=tmp_path / "delta",
            schema_dir=tmp_path / "schemas",
            quality_log_dir=tmp_path / "quality_logs",
        )

        # ------------------------------------------------------------------
        # Run the pipeline
        # ------------------------------------------------------------------
        print_section("Running Pipeline: raw → bronze → silver → gold")
        summary = pipeline.run()

        # ------------------------------------------------------------------
        # Bronze
        # ------------------------------------------------------------------
        print_section("Bronze Layer")
        bronze_df = read_delta(tmp_path / "delta" / cfg.BRONZE_TABLE)
        print(f"  Files ingested : {summary['bronze']['new_files']}")
        print(f"  Total rows     : {len(bronze_df)}")
        print(f"  Metadata cols  : _source_file, _ingested_at, _batch_id")
        print_table(
            bronze_df[["VendorID", "tpep_pickup_datetime", "fare_amount",
                        "_source_file", "_batch_id"]],
            title="Sample bronze rows:",
        )

        # ------------------------------------------------------------------
        # Silver
        # ------------------------------------------------------------------
        print_section("Silver Layer")
        silver_df = read_delta(tmp_path / "delta" / cfg.SILVER_TABLE)
        quarantine_df = read_delta(tmp_path / "delta" / cfg.SILVER_QUARANTINE_TABLE)
        print(f"  Valid rows     : {len(silver_df)}")
        print(f"  Quarantined    : {len(quarantine_df)}")
        print(f"  Row invariant  : {len(silver_df) + len(quarantine_df)} == {len(bronze_df)} (bronze)")

        if not silver_df.empty:
            print_table(
                silver_df[["pickup_at", "dropoff_at", "fare_amount",
                            "trip_duration_min", "fare_per_mile"]].head(),
                title="Sample silver rows:",
            )

        if not quarantine_df.empty:
            print_table(
                quarantine_df[["pickup_at", "fare_amount", "trip_distance",
                               "_quarantine_reason"]].head(),
                title="Quarantined rows:",
            )

        # ------------------------------------------------------------------
        # Gold
        # ------------------------------------------------------------------
        print_section("Gold Layer")
        daily_df = read_delta(tmp_path / "delta" / cfg.GOLD_DAILY_TABLE)
        hourly_df = read_delta(tmp_path / "delta" / cfg.GOLD_HOURLY_TABLE)
        payment_df = read_delta(tmp_path / "delta" / cfg.GOLD_PAYMENT_TABLE)

        print(f"  gold_daily_stats   : {len(daily_df)} rows")
        print(f"  gold_hourly_demand : {len(hourly_df)} rows")
        print(f"  gold_payment_mix   : {len(payment_df)} rows")

        if not daily_df.empty:
            print_table(daily_df, max_rows=5, title="gold_daily_stats (top 5):")

        if not payment_df.empty:
            print_table(payment_df, max_rows=6, title="gold_payment_mix:")

        # ------------------------------------------------------------------
        # Quality results
        # ------------------------------------------------------------------
        print_section("Quality Check Results")
        for key in ("bronze_quality", "bronze_silver_quality", "silver_gold_quality"):
            if key in summary:
                q = summary[key]
                status = "PASS" if q["passed"] else "FAIL"
                warn_flag = " (with warnings)" if q.get("has_warnings") else ""
                print(f"  {key:30s}: {status}{warn_flag}")

        # ------------------------------------------------------------------
        # Schema snapshot
        # ------------------------------------------------------------------
        print_section("Schema Tracker")
        schema_files = list((tmp_path / "schemas").glob("*.json"))
        print(f"  Snapshots written  : {len(schema_files)}")
        for f in schema_files:
            print(f"    {f.name}")

        # ------------------------------------------------------------------
        # Summary
        # ------------------------------------------------------------------
        print_section("Pipeline Summary")
        print(f"  Elapsed: {summary.get('elapsed_sec', '?')}s")
        print(f"  Bronze rows   : {len(bronze_df)}")
        print(f"  Silver rows   : {len(silver_df)}")
        print(f"  Quarantined   : {len(quarantine_df)}")
        print(f"  Gold daily    : {len(daily_df)}")
        print(f"  Gold hourly   : {len(hourly_df)}")
        print(f"  Gold payment  : {len(payment_df)}")
        print()
        print("  Done. All Delta tables written to temp dir (cleaned up on exit).")
        print()


if __name__ == "__main__":
    main()
