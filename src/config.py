"""
config.py — Central configuration for the Databricks Lakehouse project.

All paths, table names, quality thresholds, and schema definitions live here.
Change DELTA_DIR to point at a DBFS path when running on Databricks Community Edition.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Root paths — all relative to project root so the project is portable
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.parent

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
SAMPLE_DATA_DIR = PROJECT_ROOT / "data" / "sample"
DELTA_DIR = PROJECT_ROOT / "data" / "delta"
SCHEMA_DIR = PROJECT_ROOT / "data" / "schemas"
QUALITY_LOG_DIR = PROJECT_ROOT / "data" / "quality_logs"

# ---------------------------------------------------------------------------
# Delta table names (used as sub-directories under DELTA_DIR)
# ---------------------------------------------------------------------------
BRONZE_TABLE = "bronze_trips"
SILVER_TABLE = "silver_trips"
SILVER_QUARANTINE_TABLE = "silver_trips_quarantine"
GOLD_DAILY_TABLE = "gold_daily_stats"
GOLD_HOURLY_TABLE = "gold_hourly_demand"
GOLD_PAYMENT_TABLE = "gold_payment_mix"
QUALITY_LOG_TABLE = "data_quality_log"
SCHEMA_EVOLUTION_TABLE = "schema_evolution_log"

# ---------------------------------------------------------------------------
# TLC download config
# ---------------------------------------------------------------------------
TLC_BASE_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data"
MONTHS = ["2023-01", "2023-02", "2023-03"]

# ---------------------------------------------------------------------------
# Data quality thresholds
# ---------------------------------------------------------------------------
# Null rate: fraction of nulls allowed per column before warning/fail
MAX_NULL_RATE_WARN = 0.05   # >5%  → warn
MAX_NULL_RATE_FAIL = 0.20   # >20% → fail

# Business bounds for yellow taxi trips
FARE_MIN = 0.01        # dollars — $0 fare is invalid
FARE_MAX = 500.0       # dollars — >$500 is suspicious
DISTANCE_MIN = 0.01   # miles  — 0-distance trip is invalid
DISTANCE_MAX = 200.0  # miles  — >200 miles is suspicious
DURATION_MIN_SECONDS = 1      # seconds — instant trip is invalid
DURATION_MAX_SECONDS = 86400  # seconds — 24-hour trip is suspicious
PASSENGER_MIN = 1
PASSENGER_MAX = 9

# ---------------------------------------------------------------------------
# Expected bronze schema (column names as delivered by TLC Parquet files)
# ---------------------------------------------------------------------------
BRONZE_EXPECTED_COLUMNS = [
    "VendorID",
    "tpep_pickup_datetime",
    "tpep_dropoff_datetime",
    "passenger_count",
    "trip_distance",
    "RatecodeID",
    "store_and_fwd_flag",
    "PULocationID",
    "DOLocationID",
    "payment_type",
    "fare_amount",
    "extra",
    "mta_tax",
    "tip_amount",
    "tolls_amount",
    "improvement_surcharge",
    "total_amount",
    "congestion_surcharge",
    "airport_fee",
]

# Metadata columns added during bronze ingestion
BRONZE_METADATA_COLUMNS = ["_source_file", "_ingested_at", "_batch_id"]

# ---------------------------------------------------------------------------
# Silver column mapping: original → snake_case
# ---------------------------------------------------------------------------
SILVER_COLUMN_MAP = {
    "VendorID": "vendor_id",
    "tpep_pickup_datetime": "pickup_at",
    "tpep_dropoff_datetime": "dropoff_at",
    "passenger_count": "passenger_count",
    "trip_distance": "trip_distance",
    "RatecodeID": "rate_code_id",
    "store_and_fwd_flag": "store_and_fwd_flag",
    "PULocationID": "pickup_location_id",
    "DOLocationID": "dropoff_location_id",
    "payment_type": "payment_type_code",
    "fare_amount": "fare_amount",
    "extra": "extra_charge",
    "mta_tax": "mta_tax",
    "tip_amount": "tip_amount",
    "tolls_amount": "tolls_amount",
    "improvement_surcharge": "improvement_surcharge",
    "total_amount": "total_amount",
    "congestion_surcharge": "congestion_surcharge",
    "airport_fee": "airport_fee",
    # metadata pass-through
    "_source_file": "_source_file",
    "_ingested_at": "_ingested_at",
    "_batch_id": "_batch_id",
}

# Deduplication key for silver layer
SILVER_DEDUP_COLS = [
    "pickup_at",
    "dropoff_at",
    "pickup_location_id",
    "dropoff_location_id",
    "fare_amount",
]

# Payment type code → label mapping
PAYMENT_TYPE_LABELS = {
    1: "credit_card",
    2: "cash",
    3: "no_charge",
    4: "dispute",
    5: "unknown",
    6: "voided_trip",
}
