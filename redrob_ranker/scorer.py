"""Two-stage hybrid ranker: recall → feature rerank."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .constants import RECALL_POOL_SIZE, TITLE_RECALL_MIN_SCORE
from .disqualifiers import disqualifier_multiplier
from .embeddings import EmbeddingIndex, embeddings_available
from .features import extract_features, title_alignment_score
from .honeypot import honeypot_score
from .text_builder import build_candidate_document, tokenize


@dataclass
class RankerConfig:
    # Merit weights (sum to 1.0)
    w_title: float = 0.24
    w_narrative: float = 0.13
    w_career_narrative: float = 0.10
    w_skills: float = 0.17
    w_career: float = 0.12
    w_semantic: float = 0.14       # includes dense embedding contribution
    w_yoe: float = 0.04
    w_location: float = 0.03
    w_tail: float = 0.03

    recall_size: int = RECALL_POOL_SIZE


class TwoStageRanker:
    """
    Stage 1 — Recall top-N via BM25 (+ optional dense similarity).
    Stage 2 — Full recruiter feature fusion on recalled pool only.
    Final score = merit × availability × disqualifier × authenticity.
    """

    def __init__(
        self,
        job_description: str,
        config: RankerConfig | None = None,
        artifacts_dir: str | None = None,
    ):
        self.job_description = job_description.lower()
        self.config = config or RankerConfig()
        self.artifacts_dir = artifacts_dir
        self._jd_tokens = tokenize(self.job_description)
        self._bm25: BM25Okapi | None = None
        self._tfidf: TfidfVectorizer | None = None
        self._tfidf_matrix = None
        self._jd_tfidf = None
        self._docs: list[str] = []
        self._candidates: list[dict[str, Any]] = []
        self._embedding_index: EmbeddingIndex | None = None

    def fit(self, candidates: list[dict[str, Any]]) -> None:
        self._candidates = candidates
        self._docs = [build_candidate_document(c) for c in candidates]
        tokenized = [tokenize(d) for d in self._docs]
        self._bm25 = BM25Okapi(tokenized)

        self._tfidf = TfidfVectorizer(
            max_features=50000,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            sublinear_tf=True,
        )
        self._tfidf_matrix = self._tfidf.fit_transform(self._docs)
        self._jd_tfidf = self._tfidf.transform([self.job_description])

        if embeddings_available(self.artifacts_dir):
            self._embedding_index = EmbeddingIndex(self.artifacts_dir)

    def _semantic_scores(self, indices: np.ndarray | None = None) -> np.ndarray:
        assert self._bm25 is not None
        assert self._tfidf_matrix is not None
        assert self._jd_tfidf is not None

        if indices is None:
            indices = np.arange(len(self._candidates))

        bm25_all = np.array(self._bm25.get_scores(self._jd_tokens), dtype=np.float64)
        tfidf_all = cosine_similarity(self._jd_tfidf, self._tfidf_matrix).ravel()

        bm25 = bm25_all[indices]
        tfidf = tfidf_all[indices]
        bm25_n = (bm25 - bm25.min()) / (bm25.max() - bm25.min() + 1e-9)
        tfidf_n = (tfidf - tfidf.min()) / (tfidf.max() - tfidf.min() + 1e-9)
        lexical = 0.55 * bm25_n + 0.45 * tfidf_n

        if self._embedding_index is not None:
            dense = self._embedding_index.similarity_for_indices(indices)
            dense_n = (dense - dense.min()) / (dense.max() - dense.min() + 1e-9)
            return 0.65 * lexical + 0.35 * dense_n
        return lexical

    def _recall_indices(self) -> np.ndarray:
        assert self._bm25 is not None
        n = len(self._candidates)
        k = min(self.config.recall_size, n)

        bm25_scores = np.array(self._bm25.get_scores(self._jd_tokens), dtype=np.float64)
        top_bm25 = np.argpartition(-bm25_scores, k - 1)[:k]

        # Always include strong-title candidates (avoid missing career-narrative fits)
        title_indices = [
            i
            for i, c in enumerate(self._candidates)
            if title_alignment_score(c) >= TITLE_RECALL_MIN_SCORE
        ]

        if self._embedding_index is not None:
            dense_all = self._embedding_index.similarity_all()
            top_dense = np.argpartition(-dense_all, k - 1)[:k]
            merged = np.unique(np.concatenate([top_bm25, top_dense, np.array(title_indices)]))
        else:
            merged = np.unique(np.concatenate([top_bm25, np.array(title_indices)]))

        return merged.astype(np.int64)

    def score_all(self) -> list[tuple[str, float, dict[str, Any]]]:
        indices = self._recall_indices()
        semantic = self._semantic_scores(indices)
        cfg = self.config
        results: list[tuple[str, float, dict[str, Any]]] = []

        for local_i, idx in enumerate(indices):
            candidate = self._candidates[int(idx)]
            feats = extract_features(candidate)
            auth = honeypot_score(candidate)
            avail = feats["availability_score"]
            disq = disqualifier_multiplier(candidate)

            merit = (
                cfg.w_title * feats["title_score"]
                + cfg.w_narrative * feats["summary_narrative_score"]
                + cfg.w_career_narrative * feats["career_narrative_score"]
                + cfg.w_skills * feats["skills_score"]
                + cfg.w_career * feats["career_score"]
                + cfg.w_semantic * semantic[local_i]
                + cfg.w_yoe * feats["yoe_score"]
                + cfg.w_location * feats["location_score"]
                + cfg.w_tail * feats["tail_score"]
            )

            if feats["title_score"] < 0.15:
                merit *= 0.12
            elif feats["title_score"] < 0.4 and feats["narrative_score"] < 0.45:
                merit *= 0.3

            # Modest boost for strong career evidence without template summary
            if feats["career_narrative_score"] >= 0.7 and feats["summary_narrative_score"] < 0.5:
                merit = min(1.0, merit + 0.06)

            composite = merit * avail * disq * auth
            composite = float(max(0.0, min(1.0, composite)))

            feats["semantic_score"] = float(semantic[local_i])
            feats["authenticity"] = auth
            feats["availability_multiplier"] = avail
            feats["disqualifier_multiplier"] = disq
            feats["merit_score"] = float(merit)
            feats["composite"] = composite
            results.append((candidate["candidate_id"], composite, feats))

        return results


# Backwards-compatible alias
HybridRanker = TwoStageRanker
