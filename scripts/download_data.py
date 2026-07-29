"""
download_data.py — Download 3 months of NYC TLC yellow taxi data.

Downloads Parquet files from the TLC public dataset to data/raw/.
Also creates data/sample/ by sampling 1000 rows from month 1.

Run from project root:
    python scripts/download_data.py
"""

from pathlib import Path
import sys

import requests

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src import config as cfg


def download_file(url: str, dest: Path) -> None:
    if dest.exists():
        print(f"  Already exists: {dest.name} — skipping")
        return
    print(f"  Downloading {url} ...")
    resp = requests.get(url, stream=True, timeout=120)
    resp.raise_for_status()
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
    print(f"  Saved: {dest}")


def create_sample_from_raw(raw_file: Path, n: int = 1000) -> None:
    import pandas as pd

    sample_path = cfg.SAMPLE_DATA_DIR / f"{raw_file.stem}_sample.parquet"
    if sample_path.exists():
        print(f"Sample already exists: {sample_path}")
        return

    print(f"Sampling {n} rows from {raw_file.name} ...")
    df = pd.read_parquet(str(raw_file))
    sample = df.sample(n=min(n, len(df)), random_state=42)
    cfg.SAMPLE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    sample.to_parquet(str(sample_path), index=False)
    print(f"Sample saved: {sample_path} ({len(sample)} rows)")


def main():
    cfg.RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    first_file = None

    for month in cfg.MONTHS:
        filename = f"yellow_tripdata_{month}.parquet"
        url = f"{cfg.TLC_BASE_URL}/{filename}"
        dest = cfg.RAW_DATA_DIR / filename
        download_file(url, dest)
        if first_file is None:
            first_file = dest

    if first_file and first_file.exists():
        create_sample_from_raw(first_file)


if __name__ == "__main__":
    main()
