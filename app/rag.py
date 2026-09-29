"""RAG pipeline and vector retrieval with Qdrant (Phase 3).

Narrative match documents constructed from matches.csv and deliveries.csv,
embedded using Ollama with nomic-embed-text prefixes, and stored in Qdrant.
"""

import os
import time
import uuid
from pathlib import Path
from typing import Any
from dotenv import load_dotenv
import pandas as pd
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_ollama import OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import TokenTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.http import models

load_dotenv()

# Configuration from environment
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
MATCHES_PATH = DATA_DIR / "matches.csv"
DELIVERIES_PATH = DATA_DIR / "deliveries.csv"

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "ipl_matches")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# Bowler dismissals that count towards bowler wicket statistics
BOWLER_WICKET_KINDS = {
    "caught",
    "bowled",
    "lbw",
    "stumped",
    "caught and bowled",
    "hit wicket",
}


class NomicEmbeddingsWrapper(Embeddings):
    """Wraps OllamaEmbeddings to prepend task prefixes required by nomic-embed-text.

    Prepends 'search_document: ' when embedding documents, and
    'search_query: ' when embedding queries. Plain text without prefixes
    is stored in the Qdrant document payload.
    """

    def __init__(
        self,
        model: str = EMBED_MODEL,
        base_url: str = OLLAMA_BASE_URL,
    ):
        self._embeddings = OllamaEmbeddings(model=model, base_url=base_url)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        prefixed = [f"search_document: {t}" for t in texts]
        return self._embeddings.embed_documents(prefixed)

    def embed_query(self, text: str) -> list[float]:
        return self._embeddings.embed_query(f"search_query: {text}")


def get_embeddings() -> NomicEmbeddingsWrapper:
    """Return an instance of the prefixed NomicEmbeddingsWrapper."""
    return NomicEmbeddingsWrapper(
        model=os.getenv("EMBED_MODEL", EMBED_MODEL),
        base_url=os.getenv("OLLAMA_BASE_URL", OLLAMA_BASE_URL),
    )


def get_qdrant_client() -> QdrantClient:
    """Return a Qdrant client connection."""
    url = os.getenv("QDRANT_URL", QDRANT_URL)
    return QdrantClient(url=url)


def build_match_documents() -> list[Document]:
    """Build narrative documents for every match in matches.csv joined with deliveries.csv.

    Computes:
      - Top scorer and runs
      - Best bowler (most wickets, tiebreak by lowest economy)
      - Innings runs per team
    Composes a natural-language paragraph and chunks with a token-aware splitter.
    """
    if not MATCHES_PATH.exists() or not DELIVERIES_PATH.exists():
        raise FileNotFoundError(
            f"Dataset files missing in {DATA_DIR}. Ensure matches.csv and deliveries.csv exist."
        )

    print("📊 Loading matches.csv and deliveries.csv...")
    matches_df = pd.read_csv(MATCHES_PATH)
    deliveries_df = pd.read_csv(DELIVERIES_PATH)

    print("⚡ Precomputing match-level aggregations across 150,000+ deliveries...")

    # 1. Total runs per inning and team
    innings_summary_df = (
        deliveries_df.groupby(["match_id", "inning", "batting_team"])["total_runs"]
        .sum()
        .reset_index()
    )
    innings_by_match: dict[int, list[str]] = {}
    for _, row in innings_summary_df.iterrows():
        mid = int(row["match_id"])
        inn = int(row["inning"])
        team = str(row["batting_team"])
        runs = int(row["total_runs"])
        inn_text = f"Innings {inn} ({team}) scored {runs} runs."
        innings_by_match.setdefault(mid, []).append(inn_text)

    # 2. Batsman total runs per match
    bat_df = (
        deliveries_df.groupby(["match_id", "batsman"])["batsman_runs"]
        .sum()
        .reset_index()
    )
    top_batsmen_df = (
        bat_df.sort_values(by=["match_id", "batsman_runs"], ascending=[True, False])
        .drop_duplicates(subset=["match_id"])
        .set_index("match_id")
    )

    # 3. Bowler stats per match (wickets, runs conceded, legal balls, economy)
    bowler_w_df = (
        deliveries_df[deliveries_df["dismissal_kind"].isin(BOWLER_WICKET_KINDS)]
        .groupby(["match_id", "bowler"])["player_dismissed"]
        .count()
        .rename("wickets")
    )

    legal_balls_df = (
        deliveries_df[
            (deliveries_df["wide_runs"] == 0) & (deliveries_df["noball_runs"] == 0)
        ]
        .groupby(["match_id", "bowler"])["ball"]
        .count()
        .rename("balls")
    )

    bowler_runs_df = (
        deliveries_df.assign(
            runs_conceded=deliveries_df["batsman_runs"]
            + deliveries_df["wide_runs"]
            + deliveries_df["noball_runs"]
        )
        .groupby(["match_id", "bowler"])["runs_conceded"]
        .sum()
        .rename("runs_conceded")
    )

    bowler_stats = (
        pd.concat([legal_balls_df, bowler_runs_df, bowler_w_df], axis=1)
        .fillna({"wickets": 0, "balls": 0, "runs_conceded": 0})
        .reset_index()
    )
    bowler_stats["overs"] = bowler_stats["balls"] / 6.0
    bowler_stats["economy"] = bowler_stats.apply(
        lambda r: r["runs_conceded"] / r["overs"] if r["overs"] > 0 else 99.0, axis=1
    )

    best_bowlers_df = (
        bowler_stats.sort_values(
            by=["match_id", "wickets", "economy"], ascending=[True, False, True]
        )
        .drop_duplicates(subset=["match_id"])
        .set_index("match_id")
    )

    print("📝 Composing natural-language match summaries...")
    raw_documents: list[Document] = []

    for _, match in matches_df.iterrows():
        mid = int(match["id"])
        season = int(match["season"])
        date = str(match["date"])
        team1 = str(match["team1"])
        team2 = str(match["team2"])
        venue = str(match["venue"])
        city = match["city"]

        # City handling (Dubai matches in 2014 have missing city)
        if pd.notna(city) and str(city).strip():
            venue_desc = f"{venue}, {city}"
        elif "Dubai" in venue:
            venue_desc = f"{venue}, Dubai"
        else:
            venue_desc = venue

        # Toss summary
        toss_winner = match["toss_winner"]
        toss_decision = str(match["toss_decision"]).lower()
        if pd.notna(toss_winner):
            decision_text = "field first" if toss_decision == "field" else "bat first"
            toss_summary = f"{toss_winner} won the toss and elected to {decision_text}."
        else:
            toss_summary = ""

        # Innings summary
        inns_parts = innings_by_match.get(mid, [])
        innings_text = " ".join(inns_parts)

        # Result & Winner summary
        winner = match["winner"]
        result = str(match["result"]).lower()
        win_by_runs = int(match["win_by_runs"])
        win_by_wickets = int(match["win_by_wickets"])
        dl_applied = int(match["dl_applied"])

        if pd.isna(winner) or result == "no result":
            result_summary = "The match ended with no result (abandoned or rained out)."
        elif result == "tie":
            result_summary = (
                f"The match ended in a tie, and {winner} won in the Super Over."
            )
        else:
            if win_by_runs > 0:
                margin = f"{win_by_runs} runs"
            elif win_by_wickets > 0:
                margin = f"{win_by_wickets} wickets"
            else:
                margin = "a narrow margin"
            dl_text = " (Duckworth-Lewis method applied)" if dl_applied else ""
            result_summary = f"{winner} won by {margin}{dl_text}."

        # Player of the match
        pom = match["player_of_match"]
        pom_summary = (
            f"Player of the Match was {pom}." if pd.notna(pom) else ""
        )

        # Top scorer
        top_scorer_summary = ""
        if mid in top_batsmen_df.index:
            tb_row = top_batsmen_df.loc[mid]
            top_scorer_summary = (
                f"Top scorer was {tb_row['batsman']} with {int(tb_row['batsman_runs'])} runs."
            )

        # Best bowler
        best_bowler_summary = ""
        if mid in best_bowlers_df.index:
            bb_row = best_bowlers_df.loc[mid]
            w_count = int(bb_row["wickets"])
            econ = float(bb_row["economy"])
            best_bowler_summary = (
                f"Best bowler was {bb_row['bowler']} taking {w_count} wickets (economy: {econ:.2f})."
            )

        # Construct final narrative paragraph
        content_parts = [
            f"Match {mid} of the {season} IPL season was played on {date} at {venue_desc} between {team1} and {team2}.",
            toss_summary,
            innings_text,
            result_summary,
            pom_summary,
            top_scorer_summary,
            best_bowler_summary,
        ]
        paragraph = " ".join([p.strip() for p in content_parts if p.strip()])

        metadata = {
            "match_id": mid,
            "teams": f"{team1} vs {team2}",
            "season": season,
            "venue": venue_desc,
            "date": date,
        }
        raw_documents.append(Document(page_content=paragraph, metadata=metadata))

    # Token-aware splitting as safety net
    splitter = TokenTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunked_documents: list[Document] = []
    for doc in raw_documents:
        chunks = splitter.split_text(doc.page_content)
        for chunk_idx, chunk in enumerate(chunks):
            chunk_meta = dict(doc.metadata)
            chunk_meta["chunk_index"] = chunk_idx
            chunked_documents.append(Document(page_content=chunk, metadata=chunk_meta))

    print(
        f"✓ Built {len(raw_documents)} match summaries split into {len(chunked_documents)} chunks."
    )
    return chunked_documents


def ingest(reset: bool = False) -> None:
    """Ingest match documents into Qdrant vector database.

    Args:
        reset: If True, deletes existing collection before ingesting.
    """
    start_time = time.perf_counter()
    collection_name = os.getenv("QDRANT_COLLECTION", QDRANT_COLLECTION)
    client = get_qdrant_client()
    embeddings = get_embeddings()

    print(f"Connecting to Qdrant at {os.getenv('QDRANT_URL', QDRANT_URL)}...")

    # Collection reset handling
    if reset and client.collection_exists(collection_name):
        print(f"🔄 Resetting collection '{collection_name}'...")
        client.delete_collection(collection_name)

    # Ensure collection exists with cosine distance and 768 dimensions for nomic-embed-text
    if not client.collection_exists(collection_name):
        print(f"📦 Creating collection '{collection_name}' (vector size: 768, distance: Cosine)...")
        client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=768, distance=models.Distance.COSINE
            ),
        )

    # Build documents
    documents = build_match_documents()
    total_docs = len(documents)

    # Initialize QdrantVectorStore
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=embeddings,
    )

    # Generate deterministic point IDs based on match_id and chunk_index to prevent duplication on reruns
    ids = [
        str(
            uuid.uuid5(
                uuid.NAMESPACE_DNS,
                f"match_{d.metadata['match_id']}_chunk_{d.metadata.get('chunk_index', 0)}",
            )
        )
        for d in documents
    ]

    # Ingest in batches with progress indicator
    batch_size = 50
    print(f"🚀 Embedding and indexing {total_docs} document chunks in batches of {batch_size}...")

    for i in range(0, total_docs, batch_size):
        batch_docs = documents[i : i + batch_size]
        batch_ids = ids[i : i + batch_size]
        vector_store.add_documents(documents=batch_docs, ids=batch_ids)
        progress = min(i + batch_size, total_docs)
        pct = (progress / total_docs) * 100
        print(
            f"   Indexed [{progress}/{total_docs}] chunks ({pct:.1f}%)...",
            flush=True,
        )

    elapsed = time.perf_counter() - start_time
    print(
        f"✅ Successfully ingested {total_docs} match documents into collection '{collection_name}' in {elapsed:.2f}s."
    )


def search(query: str, k: int = 4) -> list[dict[str, Any]]:
    """Perform similarity search over IPL match documents in Qdrant.

    Args:
        query: User search query.
        k: Number of documents to retrieve (default: 4).

    Returns:
        List of dicts:
            - text: Document text
            - metadata: dict with match_id, teams, season, venue, date
            - score: Similarity score
    """
    collection_name = os.getenv("QDRANT_COLLECTION", QDRANT_COLLECTION)
    client = get_qdrant_client()
    embeddings = get_embeddings()

    if not client.collection_exists(collection_name):
        return []

    vector_store = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=embeddings,
    )

    results_with_scores = vector_store.similarity_search_with_score(query=query, k=k)

    output = []
    for doc, score in results_with_scores:
        output.append(
            {
                "text": doc.page_content,
                "metadata": {
                    "match_id": doc.metadata.get("match_id"),
                    "teams": doc.metadata.get("teams"),
                    "season": doc.metadata.get("season"),
                    "venue": doc.metadata.get("venue"),
                    "date": doc.metadata.get("date"),
                },
                "score": float(score),
            }
        )

    return output
