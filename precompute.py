#!/usr/bin/env python3
"""
Offline precomputation — allowed to exceed 5 minutes and use network once.

Usage:
    python precompute.py --candidates ./candidates.jsonl --jd ./job_description.txt
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from redrob_ranker.embeddings import precompute_embeddings
from redrob_ranker.loader import load_candidates, load_job_description


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Precompute candidate embeddings")
    parser.add_argument("--candidates", required=True, type=Path)
    parser.add_argument("--jd", type=Path, default=None)
    parser.add_argument(
        "--artifacts",
        type=Path,
        default=Path("artifacts"),
        help="Output directory for embedding artifacts",
    )
    args = parser.parse_args(argv)

    t0 = time.perf_counter()
    jd = load_job_description(args.jd)
    candidates = load_candidates(args.candidates)
    print(f"Encoding {len(candidates):,} candidates...", flush=True)
    out = precompute_embeddings(candidates, jd, artifacts_dir=args.artifacts)
    print(f"Artifacts written to {out} in {time.perf_counter() - t0:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
