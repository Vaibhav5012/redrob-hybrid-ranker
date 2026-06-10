"""Offline embedding precomputation and fast online loading."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .constants import DEFAULT_ARTIFACTS_DIR, EMBEDDING_BATCH_SIZE, EMBEDDING_MODEL
from .text_builder import build_candidate_document


def artifacts_paths(artifacts_dir: Path | str | None = None) -> dict[str, Path]:
    base = Path(artifacts_dir) if artifacts_dir else DEFAULT_ARTIFACTS_DIR
    return {
        "dir": base,
        "embeddings": base / "candidate_embeddings.npy",
        "ids": base / "candidate_ids.json",
        "jd_embedding": base / "jd_embedding.npy",
        "meta": base / "meta.json",
    }


def embeddings_available(artifacts_dir: Path | str | None = None) -> bool:
    paths = artifacts_paths(artifacts_dir)
    return paths["embeddings"].exists() and paths["ids"].exists()


def precompute_embeddings(
    candidates: list[dict[str, Any]],
    job_description: str,
    artifacts_dir: Path | str | None = None,
    model_name: str = EMBEDDING_MODEL,
) -> Path:
    """
    Offline step — may use network to download the model once.
  Writes:
    artifacts/candidate_embeddings.npy  (N x D float32)
    artifacts/candidate_ids.json
    artifacts/jd_embedding.npy          (D,)
    artifacts/meta.json
    """
    from sentence_transformers import SentenceTransformer

    paths = artifacts_paths(artifacts_dir)
    paths["dir"].mkdir(parents=True, exist_ok=True)

    model = SentenceTransformer(model_name)
    docs = [build_candidate_document(c) for c in candidates]
    ids = [c["candidate_id"] for c in candidates]

    vectors: list[np.ndarray] = []
    for start in range(0, len(docs), EMBEDDING_BATCH_SIZE):
        batch = docs[start : start + EMBEDDING_BATCH_SIZE]
        emb = model.encode(batch, show_progress_bar=False, normalize_embeddings=True)
        vectors.append(np.asarray(emb, dtype=np.float32))

    matrix = np.vstack(vectors)
    jd_vec = model.encode([job_description], normalize_embeddings=True)[0].astype(np.float32)

    np.save(paths["embeddings"], matrix)
    np.save(paths["jd_embedding"], jd_vec)
    paths["ids"].write_text(json.dumps(ids), encoding="utf-8")
    paths["meta"].write_text(
        json.dumps({"model": model_name, "count": len(ids), "dim": matrix.shape[1]}),
        encoding="utf-8",
    )
    return paths["dir"]


class EmbeddingIndex:
    """Load precomputed embeddings and compute JD similarity."""

    def __init__(self, artifacts_dir: Path | str | None = None):
        paths = artifacts_paths(artifacts_dir)
        if not embeddings_available(paths["dir"]):
            raise FileNotFoundError(
                f"Embeddings not found in {paths['dir']}. Run: python precompute.py"
            )
        self._matrix = np.load(paths["embeddings"])
        self._ids: list[str] = json.loads(paths["ids"].read_text(encoding="utf-8"))
        self._id_to_idx = {cid: i for i, cid in enumerate(self._ids)}
        jd_path = paths["jd_embedding"]
        self._jd = np.load(jd_path) if jd_path.exists() else None

    @property
    def ids(self) -> list[str]:
        return self._ids

    def similarity_for_indices(self, indices: np.ndarray) -> np.ndarray:
        if self._jd is None:
            return np.zeros(len(indices), dtype=np.float64)
        sims = self._matrix[indices] @ self._jd
        return sims.astype(np.float64)

    def similarity_all(self) -> np.ndarray:
        if self._jd is None:
            return np.zeros(len(self._ids), dtype=np.float64)
        return (self._matrix @ self._jd).astype(np.float64)

    def index_of(self, candidate_id: str) -> int | None:
        return self._id_to_idx.get(candidate_id)
