"""Inspect Kaggle IPL dataset (matches.csv and deliveries.csv).

Prints shape, columns, dtypes, sample rows, and null counts for both CSVs.
"""

from pathlib import Path
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
MATCHES_PATH = DATA_DIR / "matches.csv"
DELIVERIES_PATH = DATA_DIR / "deliveries.csv"


def inspect_csv(name: str, path: Path):
    print("=" * 80)
    print(f"DATASET: {name} ({path})")
    print("=" * 80)

    if not path.exists():
        print(f"ERROR: File not found at {path}")
        return

    df = pd.read_csv(path)

    print(f"\n1. SHAPE: {df.shape[0]} rows, {df.shape[1]} columns\n")

    print("2. COLUMNS & DTYPES:")
    print("-" * 50)
    for col, dtype in df.dtypes.items():
        print(f"  - {col}: {dtype}")

    print("\n3. NULL COUNTS PER COLUMN:")
    print("-" * 50)
    null_counts = df.isnull().sum()
    for col, cnt in null_counts.items():
        pct = (cnt / len(df)) * 100
        print(f"  - {col}: {cnt} nulls ({pct:.1f}%)")

    print("\n4. SAMPLE 3 ROWS:")
    print("-" * 50)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)
    print(df.head(3))
    print("\n" + "=" * 80 + "\n")


def main():
    inspect_csv("matches.csv", MATCHES_PATH)
    inspect_csv("deliveries.csv", DELIVERIES_PATH)


if __name__ == "__main__":
    main()
