"""Inspect RAG vector retrieval without making LLM calls (Phase 3).

Usage:
    python -m scripts.inspect_retrieval "Who won the 2016 IPL final?"
    python -m scripts.inspect_retrieval --query "highest scoring match" --k 5
"""

import argparse
import sys
from app.rag import search


def main():
    parser = argparse.ArgumentParser(
        description="Inspect top-k RAG vector search results from Qdrant."
    )
    parser.add_argument(
        "query_pos",
        nargs="?",
        default=None,
        help="Search query string (positional).",
    )
    parser.add_argument(
        "--query",
        "-q",
        default=None,
        help="Search query string (flag).",
    )
    parser.add_argument(
        "--k",
        "-k",
        type=int,
        default=5,
        help="Number of retrieved documents (default: 5).",
    )
    args = parser.parse_args()

    query = args.query or args.query_pos
    if not query:
        query = input("Enter search query: ").strip()

    if not query:
        print("Error: Empty query provided.")
        sys.exit(1)

    print("=" * 70)
    print(f"🔍 Searching Qdrant for: '{query}' (top {args.k} results)")
    print("=" * 70)

    results = search(query=query, k=args.k)

    if not results:
        print("No documents found in Qdrant. Did you run 'python -m scripts.ingest'?")
        return

    for rank, item in enumerate(results, 1):
        meta = item.get("metadata", {})
        score = item.get("score", 0.0)
        text = item.get("text", "")
        print(f"\n[{rank}] Score: {score:.4f} | Match ID: {meta.get('match_id')} | Season: {meta.get('season')}")
        print(f"    Teams: {meta.get('teams')} | Venue: {meta.get('venue')}")
        print(f"    Document:\n    {text}")
        print("-" * 70)


if __name__ == "__main__":
    main()
