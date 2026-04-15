"""
pipeline.py — LakehousePipeline: orchestrates bronze → silver → gold.

Runs quality checks between each layer transition.
Returns a comprehensive summary with row counts, quality results, and timing.
"""

import time
from pathlib import Path
from typing import Optional

import pandas as pd

from . import config as cfg
from .bronze import BronzeIngester
from .gold import GoldAggregator
from .quality import (
    QualityRunner,
    freshness_check,
    null_rate_check,
    row_count_match,
    uniqueness_check,
)
from .schema_tracker import SchemaEvolutionTracker
from .silver import SilverTransformer
from .utils import get_logger, read_delta

logger = get_logger(__name__)


class LakehousePipeline:
    """
    End-to-end medallion lakehouse pipeline.

    Usage:
        pipeline = LakehousePipeline()
        summary = pipeline.run(source_dir=Path("data/sample"))
    """

    def __init__(
        self,
        raw_dir: Optional[Path] = None,
        delta_dir: Optional[Path] = None,
        schema_dir: Optional[Path] = None,
        quality_log_dir: Optional[Path] = None,
    ):
        self.raw_dir = Path(raw_dir) if raw_dir else cfg.RAW_DATA_DIR
        self.delta_dir = Path(delta_dir) if delta_dir else cfg.DELTA_DIR
        self.schema_dir = Path(schema_dir) if schema_dir else cfg.SCHEMA_DIR
        self.quality_log_dir = Path(quality_log_dir) if quality_log_dir else cfg.QUALITY_LOG_DIR

        self.bronze = BronzeIngester(raw_dir=self.raw_dir, delta_dir=self.delta_dir)
        self.silver = SilverTransformer(delta_dir=self.delta_dir)
        self.gold = GoldAggregator(delta_dir=self.delta_dir)
        self.tracker = SchemaEvolutionTracker(schema_dir=self.schema_dir)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, source_dir: Optional[Path] = None) -> dict:
        """
        Full pipeline: raw Parquet → bronze → silver → gold.

        Args:
            source_dir: Directory containing raw Parquet files.
                        Defaults to config.RAW_DATA_DIR.

        Returns:
            Summary dict with row counts, quality statuses, and elapsed seconds.
        """
        if source_dir:
            self.bronze.raw_dir = Path(source_dir)

        t_start = time.time()
        summary: dict = {}

        # ---------------------------------------------------------------
        # Phase 1: Bronze ingestion
        # ---------------------------------------------------------------
        logger.info("=== Phase 1: Bronze Ingestion ===")
        bronze_summary = self.bronze.ingest_all()
        summary["bronze"] = bronze_summary

        bronze_df = read_delta(self.bronze.bronze_path)
        if bronze_df.empty:
            logger.warning("Bronze table is empty — nothing to process")
            summary["elapsed_sec"] = round(time.time() - t_start, 2)
            return summary

        # Schema tracking for bronze
        bronze_changes = self.tracker.detect_changes(bronze_df, cfg.BRONZE_TABLE)
        self.tracker.snapshot_schema(bronze_df, cfg.BRONZE_TABLE)
        if bronze_changes:
            self.tracker.log_changes(cfg.BRONZE_TABLE, bronze_changes)

        # Quality checks on bronze
        logger.info("=== Quality: Bronze ===")
        bronze_quality = self._run_bronze_quality(bronze_df)
        summary["bronze_quality"] = {
            "passed": bronze_quality.passed,
            "has_warnings": bronze_quality.has_warnings,
            "results": [r.status for r in bronze_quality.results],
        }
        self._log_quality_summary(bronze_quality)

        # ---------------------------------------------------------------
        # Phase 2: Silver transformation
        # ---------------------------------------------------------------
        logger.info("=== Phase 2: Silver Transformation ===")
        silver_summary = self.silver.transform(bronze_df)
        summary["silver"] = silver_summary

        silver_df = read_delta(self.silver.silver_path)
        quarantine_df = read_delta(self.silver.quarantine_path)

        # Schema tracking for silver
        if not silver_df.empty:
            silver_changes = self.tracker.detect_changes(silver_df, cfg.SILVER_TABLE)
            self.tracker.snapshot_schema(silver_df, cfg.SILVER_TABLE)
            if silver_changes:
                self.tracker.log_changes(cfg.SILVER_TABLE, silver_changes)

        # Quality checks: bronze → silver
        logger.info("=== Quality: Bronze → Silver ===")
        bronze_silver_quality = self._run_bronze_silver_quality(
            bronze_df, silver_df, quarantine_df, silver_summary
        )
        summary["bronze_silver_quality"] = {
            "passed": bronze_silver_quality.passed,
            "has_warnings": bronze_silver_quality.has_warnings,
            "results": [r.status for r in bronze_silver_quality.results],
        }
        self._log_quality_summary(bronze_silver_quality)

        # ---------------------------------------------------------------
        # Phase 3: Gold aggregation
        # ---------------------------------------------------------------
        logger.info("=== Phase 3: Gold Aggregation ===")
        if silver_df.empty:
            logger.warning("Silver table is empty — skipping gold build")
            summary["gold"] = {"skipped": True}
        else:
            gold_summary = self.gold.build_all(silver_df)
            summary["gold"] = gold_summary

            # Load gold tables for quality checks
            daily_df = read_delta(self.gold.daily_path)
            hourly_df = read_delta(self.gold.hourly_path)

            # Quality checks: silver → gold
            logger.info("=== Quality: Silver → Gold ===")
            silver_gold_quality = self._run_silver_gold_quality(silver_df, daily_df)
            summary["silver_gold_quality"] = {
                "passed": silver_gold_quality.passed,
                "has_warnings": silver_gold_quality.has_warnings,
                "results": [r.status for r in silver_gold_quality.results],
            }
            self._log_quality_summary(silver_gold_quality)

        summary["elapsed_sec"] = round(time.time() - t_start, 2)
        logger.info("Pipeline complete in %.1fs", summary["elapsed_sec"])
        return summary

    def run_incremental(self, new_files: list[Path]) -> dict:
        """
        Ingest only specified new Parquet files and recompute silver/gold.

        Args:
            new_files: List of Parquet file paths to ingest.

        Returns:
            Summary dict.
        """
        logger.info("Incremental run: %d new file(s)", len(new_files))

        # Temporarily redirect raw_dir to parent of first file
        if new_files:
            for path in new_files:
                self.bronze.ingest_file(path)

        # Recompute silver + gold from full bronze table
        bronze_df = read_delta(self.bronze.bronze_path)
        if bronze_df.empty:
            return {"warning": "Bronze table empty after incremental ingest"}

        silver_summary = self.silver.transform(bronze_df)
        silver_df = read_delta(self.silver.silver_path)

        gold_summary = {}
        if not silver_df.empty:
            gold_summary = self.gold.build_all(silver_df)

        return {
            "new_files_ingested": len(new_files),
            "silver": silver_summary,
            "gold": gold_summary,
        }

    # ------------------------------------------------------------------
    # Quality check suites
    # ------------------------------------------------------------------

    def _run_bronze_quality(self, bronze_df: pd.DataFrame):
        checks = [
            null_rate_check("_source_file"),
            null_rate_check("tpep_pickup_datetime", warn_threshold=0.02, fail_threshold=0.10),
            freshness_check("tpep_pickup_datetime", min_rows=1),
        ]
        runner = QualityRunner(checks, source_table="raw", target_table=cfg.BRONZE_TABLE)
        summary = runner.run_all(bronze_df)
        runner.log_results(summary, self.quality_log_dir)
        return summary

    def _run_bronze_silver_quality(
        self,
        bronze_df: pd.DataFrame,
        silver_df: pd.DataFrame,
        quarantine_df: pd.DataFrame,
        silver_summary: dict,
    ):
        # Row count invariant: silver + quarantine + deduped == bronze (no silent drops)
        # Deduplication intentionally removes exact duplicates — these are counted
        # separately in silver_summary["deduped_rows"].
        deduped = silver_summary.get("deduped_rows", 0)
        accounted_for = len(silver_df) + len(quarantine_df) + deduped
        expected = len(bronze_df)

        from .quality import QualityCheck

        def count_check_fn(df):
            if accounted_for == expected:
                return accounted_for, "pass", (
                    f"Row accounting: {len(silver_df)} silver + "
                    f"{len(quarantine_df)} quarantine + "
                    f"{deduped} deduped = {accounted_for} == {expected} bronze"
                )
            diff = accounted_for - expected
            return accounted_for, "fail", (
                f"Row count mismatch: accounted={accounted_for}, bronze={expected} (diff={diff:+d})"
            )

        checks = [
            QualityCheck("silver+quarantine+deduped==bronze", count_check_fn),
            null_rate_check("pickup_at", warn_threshold=0.02, fail_threshold=0.10),
            null_rate_check("fare_amount", warn_threshold=0.02, fail_threshold=0.10),
            freshness_check("pickup_at", min_rows=1),
        ]

        # Run non-count checks against silver (combined with quarantine for null checks)
        combined_df = pd.concat([silver_df, quarantine_df], ignore_index=True)
        runner = QualityRunner(checks, source_table=cfg.BRONZE_TABLE, target_table=cfg.SILVER_TABLE)
        summary = runner.run_all(combined_df)
        runner.log_results(summary, self.quality_log_dir)
        return summary

    def _run_silver_gold_quality(
        self,
        silver_df: pd.DataFrame,
        daily_df: pd.DataFrame,
    ):
        checks = [
            uniqueness_check(["trip_date", "pickup_zone"]),
            freshness_check("trip_date", min_rows=1),
        ]
        runner = QualityRunner(checks, source_table=cfg.SILVER_TABLE, target_table=cfg.GOLD_DAILY_TABLE)
        summary = runner.run_all(daily_df)
        runner.log_results(summary, self.quality_log_dir)
        return summary

    def _log_quality_summary(self, summary) -> None:
        logger.info(str(summary))
