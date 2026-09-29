"""CLI script to ingest Kaggle IPL match documents into Qdrant (Phase 3).

Usage:
    python -m scripts.ingest [--reset]
"""

import argparse
from app.rag import ingest


def main():
    parser = argparse.ArgumentParser(
        description="Ingest IPL match summaries into Qdrant vector database."
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Reset and recreate the Qdrant collection before ingestion.",
    )
    args = parser.parse_args()

    ingest(reset=args.reset)


if __name__ == "__main__":
    main()
