# Databricks Medallion Lakehouse

A portfolio data engineering project demonstrating medallion architecture (Bronze → Silver → Gold) with Delta Lake, built to run locally via the `deltalake` Python library and deployable to Databricks Community Edition.

## Dataset

NYC Taxi & Limousine Commission (TLC) yellow taxi trip records. 3 months of 2023 data (~10M rows). Freely available at https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page.

Rich schema (timestamps, locations, fares, tips, payment types) with real-world data quality issues — nulls, zero fares, impossible distances — ideal for demonstrating a production-grade data quality pipeline.

## Architecture

```
Raw Parquet (data/raw/)
        │
        ▼
┌─────────────────────┐
│   BRONZE LAYER      │  Append-only. Preserves original schema.
│   bronze_trips      │  Adds: _source_file, _ingested_at, _batch_id
└─────────────────────┘
        │   ← quality: null rates, schema validation
        ▼
┌─────────────────────┐
│   SILVER LAYER      │  Cleaned, typed, validated, deduplicated.
│   silver_trips      │  Invalid rows → silver_trips_quarantine
│   silver_quarantine │  (never silently dropped)
└─────────────────────┘
        │   ← quality: row count invariant, freshness, ranges
        ▼
┌─────────────────────┐
│   GOLD LAYER        │  Business-level aggregates
│   gold_daily_stats  │  Per (date, pickup_zone): counts, revenue, avg fare
│   gold_hourly_demand│  Per (date, hour, zone): demand heatmap data
│   gold_payment_mix  │  Per (date, payment_type): pct distribution
└─────────────────────┘
```

## Quick Start (local, no Databricks needed)

```bash
git clone <repo>
cd databricks-lakehouse

python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Generate the 300-row synthetic sample (already committed, but can regenerate)
python scripts/create_sample.py

# Run the full pipeline on the sample data
python examples/quick_demo.py
```

## Download Full Dataset (optional)

```bash
# Downloads 3 months of TLC yellow taxi Parquet to data/raw/
python scripts/download_data.py
```

## Running Tests

```bash
pytest tests/ -v
```

All 40+ tests run without Spark, without downloads, and without external API calls.

## Project Structure

```
src/
  config.py         — paths, table names, quality thresholds, schema definitions
  bronze.py         — BronzeIngester: raw Parquet → bronze Delta table
  silver.py         — SilverTransformer: bronze → silver + quarantine
  gold.py           — GoldAggregator: silver → gold aggregate tables
  quality.py        — QualityCheck, QualityRunner (reusable framework)
  schema_tracker.py — SchemaEvolutionTracker: detect and log schema drift
  pipeline.py       — LakehousePipeline: end-to-end orchestration
  utils.py          — Delta table helpers, logging, date utilities

tests/              — 40+ tests, no Spark required
data/sample/        — Committed 300-row synthetic sample (runs immediately after clone)
data/schemas/       — Schema snapshots (JSON) written at runtime
notebooks/          — Databricks Community Edition exploration notebook
scripts/
  create_sample.py  — Generate synthetic sample data
  download_data.py  — Download full 3-month TLC dataset
```

## Data Quality Framework

Quality checks run between every layer transition. Results are logged to `data/quality_logs/`.

| Check | Between | Rule |
|---|---|---|
| Null rate | Raw → Bronze | Null rate per column < configurable threshold |
| Row count match | Bronze → Silver | `silver + quarantine == bronze` (no data loss) |
| Range validation | Bronze → Silver | Fare, distance, duration within expected bounds |
| Freshness | Silver → Gold | Silver has non-null pickup timestamps |
| Uniqueness | Silver → Gold | No duplicate (date, zone) keys in gold tables |

The `QualityCheck` and `QualityRunner` classes are reusable — not ad-hoc assertions. Each check returns a `QualityResult` with pass/warn/fail status and the raw metric value.

## Schema Evolution Tracking

`SchemaEvolutionTracker` compares each layer's schema against the previous run's JSON snapshot. Detected changes (added, removed, type-changed columns) are:
- Logged to `data/schemas/schema_evolution_log.jsonl`
- Printed as warnings — does not halt the pipeline
- Reviewed by a human before promotion

This mirrors production Databricks workflows where upstream schemas evolve.

## Running on Databricks Community Edition

1. Upload `data/raw/` Parquet files to DBFS (`/FileStore/`)
2. Import `notebooks/exploration.ipynb`
3. Update `DELTA_DIR` in `src/config.py` to point at a DBFS path
4. Delta Lake is built into Databricks — no extra setup needed

## Key Design Decisions

- **deltalake (not Spark) for local dev**: All tests and demos run on a laptop without Spark installed. Same logic runs on Databricks via PySpark Delta.
- **Quarantine over silent drops**: Invalid rows are never deleted — they go to `silver_trips_quarantine` with a `_quarantine_reason` column. Row count invariant holds: `bronze == silver + quarantine`.
- **Three gold tables over one wide table**: Each serves a specific analytical purpose (daily dashboards, demand heatmaps, payment trends).
- **Schema evolution warnings, not errors**: Real pipelines encounter upstream schema drift. Tracking it without blocking lets the pipeline continue while flagging changes for human review.

## Requirements

```
pandas>=2.0
numpy>=1.24
pyarrow>=14.0
deltalake>=0.14
pytest>=7.0
requests>=2.31
```
