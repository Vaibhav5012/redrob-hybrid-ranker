#!/usr/bin/env python3
"""
Redrob AI Recruiter v2 — two-stage hybrid ranker.

Usage:
    # Step 1 (offline, once — may use network to download embedding model):
    python precompute.py --candidates ./candidates.jsonl

    # Step 2 (online ranking — CPU only, no network, <5 min):
    python rank.py --candidates ./candidates.jsonl --out ./submission.csv
    python rank.py --candidates ./candidates.jsonl --out ./submission.csv --no-embeddings
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

from redrob_ranker.embeddings import embeddings_available
from redrob_ranker.loader import load_candidates, load_job_description
from redrob_ranker.reasoning import build_reasoning
from redrob_ranker.scorer import TwoStageRanker


def write_submission(
    ranked: list[tuple[str, float, dict]],
    out_path: Path,
    top_n: int = 100,
) -> None:
    top = ranked[:top_n]
    if len(top) < top_n:
        raise ValueError(f"Only {len(top)} candidates available; need {top_n}")

    raw_scores = [s for _, s, _ in top]
    max_s, min_s = max(raw_scores), min(raw_scores)
    span = max(max_s - min_s, 1e-9)

    rows = []
    for rank, (cid, score, feats) in enumerate(top, start=1):
        normalized = 0.2 + 0.79 * (score - min_s) / span
        rank_floor = 0.99 - 0.0079 * (rank - 1)
        final_score = min(normalized, rank_floor)
        rows.append(
            {
                "candidate_id": cid,
                "rank": rank,
                "score": round(final_score, 4),
                "reasoning": build_reasoning(feats),
            }
        )

    for i in range(1, len(rows)):
        if rows[i]["score"] > rows[i - 1]["score"]:
            rows[i]["score"] = rows[i - 1]["score"]
        if (
            rows[i]["score"] == rows[i - 1]["score"]
            and rows[i]["candidate_id"] < rows[i - 1]["candidate_id"]
        ):
            rows[i]["score"] = round(rows[i - 1]["score"] - 0.0001, 4)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["candidate_id", "rank", "score", "reasoning"]
        )
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Rank candidates for Redrob JD (v2)")
    parser.add_argument("--candidates", required=True, type=Path)
    parser.add_argument("--jd", type=Path, default=None)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--top", type=int, default=100)
    parser.add_argument(
        "--artifacts",
        type=Path,
        default=Path("artifacts"),
        help="Directory with precomputed embeddings",
    )
    parser.add_argument(
        "--no-embeddings",
        action="store_true",
        help="Skip dense embeddings even if artifacts exist",
    )
    args = parser.parse_args(argv)

    t0 = time.perf_counter()
    jd = load_job_description(args.jd)
    candidates = load_candidates(args.candidates)
    print(f"Loaded {len(candidates):,} candidates", flush=True)

    artifacts_dir = None if args.no_embeddings else str(args.artifacts)
    if artifacts_dir and embeddings_available(artifacts_dir):
        print(f"Using precomputed embeddings from {artifacts_dir}", flush=True)
    else:
        print("Running without dense embeddings (BM25+TF-IDF only)", flush=True)
        artifacts_dir = None

    ranker = TwoStageRanker(jd, artifacts_dir=artifacts_dir)
    ranker.fit(candidates)
    scored = ranker.score_all()
    scored.sort(key=lambda x: (-x[1], x[0]))

    write_submission(scored, args.out, top_n=args.top)
    elapsed = time.perf_counter() - t0
    print(f"Wrote top {args.top} to {args.out} in {elapsed:.1f}s", flush=True)

    print("\nTop 10 preview:")
    for rank, (cid, score, feats) in enumerate(scored[:10], start=1):
        print(
            f"  {rank:2d}. {cid}  score={score:.4f}  "
            f"{feats['title']} ({feats['yoe']:.1f}y)  "
            f"avail={feats['availability_multiplier']:.2f}  "
            f"career_ir={feats['career_narrative_score']:.2f}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
