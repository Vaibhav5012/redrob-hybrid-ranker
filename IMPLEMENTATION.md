# Redrob Ranker v2 — Full Implementation Guide

This document explains **how every improvement is implemented**, how the pieces connect, and the exact commands to run. The code in this repo already follows this design.

---

## Architecture overview

```
┌─────────────────────────────────────────────────────────────────┐
│  OFFLINE (precompute.py) — network OK, can take >5 min          │
│  • Download sentence-transformers model once                      │
│  • Encode 100K candidate docs → artifacts/candidate_embeddings.npy│
│  • Encode JD → artifacts/jd_embedding.npy                         │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│  ONLINE (rank.py) — CPU only, no network, ≤5 min                  │
│                                                                   │
│  Stage 1: RECALL (~5K candidates)                                 │
│    • BM25 top 5000                                                │
│    • Dense embedding top 5000 (if artifacts exist)                │
│    • Union: all title_score ≥ 0.72 (never miss strong titles)     │
│                                                                   │
│  Stage 2: RERANK (recalled pool only)                             │
│    merit = weighted features (title, skills, career IR, …)        │
│    final = merit × availability × disqualifiers × authenticity    │
│                                                                   │
│  Output: submission.csv (top 100 + reasoning)                     │
└─────────────────────────────────────────────────────────────────┘
```

**Why two stages?** Scoring 100K profiles with rich features is fast, but recall quality matters. Stage 1 casts a wide net; Stage 2 applies recruiter judgment only where it counts. This improves **NDCG@50** and **MAP** without blowing the 5-minute budget.

---

## Step-by-step setup

### 1. Install dependencies

```bash
cd Redrob
pip install -r requirements.txt
```

### 2. Offline — precompute embeddings (recommended)

```bash
python precompute.py \
  --candidates "./[PUB] India_runs_data_and_ai_challenge/[PUB] India_runs_data_and_ai_challenge/India_runs_data_and_ai_challenge/candidates.jsonl" \
  --jd ./job_description.txt \
  --artifacts ./artifacts
```

**What happens internally** (`redrob_ranker/embeddings.py`):

1. Loads `sentence-transformers/all-MiniLM-L6-v2` (downloads once).
2. Flattens each candidate via `build_candidate_document()` (headline, summary, career, skills).
3. Encodes in batches of 256 → `float32` matrix `(100000, 384)`.
4. Saves:
   - `artifacts/candidate_embeddings.npy`
   - `artifacts/candidate_ids.json`
   - `artifacts/jd_embedding.npy`
   - `artifacts/meta.json`

**Disk:** ~150 MB. **Time:** ~10–20 min first run on CPU.

### 3. Online — produce submission

```bash
python rank.py \
  --candidates "./[PUB] India_runs_data_and_ai_challenge/[PUB] India_runs_data_and_ai_challenge/India_runs_data_and_ai_challenge/candidates.jsonl" \
  --jd ./job_description.txt \
  --out ./submission.csv \
  --artifacts ./artifacts
```

Skip embeddings (faster, slightly lower quality):

```bash
python rank.py --candidates ./candidates.jsonl --out ./submission.csv --no-embeddings
```

### 4. Validate

```bash
python "./[PUB] India_runs_data_and_ai_challenge/[PUB] India_runs_data_and_ai_challenge/India_runs_data_and_ai_challenge/validate_submission.py" submission.csv
```

---

## Improvement 1: Behavioral re-ranking in top tier

**Problem:** Template-matched profiles with 10% recruiter response rate were ranking above engaged India-based candidates.

**Implementation:** `availability_score()` in `redrob_ranker/features.py`

| Signal | Effect |
|--------|--------|
| `recruiter_response_rate` | `× (0.55 + 0.45 × rate)`; extra `×0.72` if `< 0.2` |
| `open_to_work_flag` | `×0.82` if false |
| `last_active_date` | `×0.65` if inactive > 1 year |
| `notice_period_days` | Penalty above 90 days |
| `interview_completion_rate` | Mild boost |

**Fusion in scorer:**

```python
composite = merit * availability * disqualifier * authenticity
```

Availability is a **multiplier**, not a small tweak — it directly reorders elite candidates.

---

## Improvement 2: Career-history IR scoring (anti template overfit)

**Problem:** `planted_narrative_score` only matched exact summary phrases all teams will find.

**Implementation:** `career_narrative_score()` in `features.py`

- Scans **career `description` + `title`** fields for markers in `CAREER_IR_MARKERS` (`constants.py`): `hybrid retrieval`, `ndcg`, `bm25`, `p95`, `embedding drift`, etc.
- `combined_narrative_score()` = `max(summary, 0.55×summary + 0.45×career)`
- Scorer weights: `w_narrative=0.14` (summary), `w_career_narrative=0.10` (career)
- Bonus: `+0.06` merit if `career_narrative ≥ 0.7` and summary template absent

**Files:** `constants.py` → `CAREER_IR_MARKERS`, `features.py` → `career_narrative_score()`

---

## Improvement 3: JD disqualifier penalties

**Problem:** AI Research Engineers, consulting-only profiles, CV-without-IR, LangChain tourists still ranked too high.

**Implementation:** `redrob_ranker/disqualifiers.py`

| Function | Penalty | JD basis |
|----------|---------|----------|
| `research_penalty()` | ×0.25–0.45 | "pure research without production" |
| `consulting_only_penalty()` | ×0.5–0.75 | TCS/Infosys/Wipro-only careers |
| `cv_speech_without_ir_penalty()` | ×0.4–0.55 | CV/speech without NLP/IR |
| `langchain_tourist_penalty()` | ×0.55 | LangChain-only, shallow ML |
| `career_consistency_penalty()` | ×0.2–0.8 | AI title + non-tech role descriptions |

Combined: `disqualifier_multiplier()` — product of all, floor 0.08.

**Wiring:** `scorer.py` → `composite *= disqualifier_multiplier(candidate)`

---

## Improvement 4: Two-stage recall → rerank

**Implementation:** `TwoStageRanker` in `redrob_ranker/scorer.py`

### Stage 1 — `_recall_indices()`

```python
top_bm25 = argpartition(bm25_scores, 5000)
top_dense = argpartition(embedding_sims, 5000)   # if artifacts exist
title_must = [i for i, c in enumerate(cands) if title_score(c) >= 0.72]
return unique(concat(top_bm25, top_dense, title_must))
```

**Why union with title_must?** A Search Engineer with great career IR might have low BM25 if they don't repeat JD buzzwords in summary.

### Stage 2 — `score_all()`

Only recalled indices get full feature fusion. Sort by `composite`, tie-break `candidate_id` ascending.

**Config:** `RECALL_POOL_SIZE = 5000` in `constants.py`, override via `RankerConfig(recall_size=...)`.

---

## Improvement 5: Precomputed dense embeddings

**Implementation:** `redrob_ranker/embeddings.py` + `precompute.py`

| Phase | Network | What runs |
|-------|---------|-----------|
| Precompute | Allowed | `SentenceTransformer.encode()` |
| Rank | **Forbidden** | `np.load()` + dot product |

Semantic blend in `_semantic_scores()`:

```python
lexical = 0.55 * bm25_norm + 0.45 * tfidf_norm
if embeddings:
    return 0.65 * lexical + 0.35 * dense_norm
```

**Hackathon compliance:** `has_network_during_ranking: false` — ranking only reads `.npy` files.

---

## Improvement 6: Tail differentiation (ranks 50–100)

**Problem:** Scores flat ~0.575 at cutoff — bad for MAP.

**Implementation:** `tail_signals_score()` in `features.py`

- `saved_by_recruiters_30d`
- `profile_views_received_30d`
- `github_activity_score`
- `skill_assessment_scores` (NLP/ML/retrieval)
- Education tier + CS field
- `profile_completeness_score`

Weight in merit: `w_tail = 0.04` — small but breaks ties among similar ML engineers.

---

## Improvement 7: Stronger honeypot detection

**Implementation:** `redrob_ranker/honeypot.py` (unchanged core) + `career_consistency_penalty()` in disqualifiers

Honeypot signals:
- Expert skills @ 0 months (≥3)
- Timeline overlap (career months >> YOE)
- AI title + 2+ non-tech role descriptions
- Buzzword skills with <3 months duration

**Audit script:** `python analyze_gaps.py` — check top 20 for flags.

---

## Improvement 8: Reasoning for Stage 4 review

**Implementation:** `redrob_ranker/reasoning.py`

Each row is **unique** and includes:
- Current title + company + YOE
- Prior employers (from career history)
- Named skills (from profile, not hallucinated)
- JD hook ("aligns with JD need for production retrieval/ranking…")
- Honest concerns (low response rate, notice period, non-India, above YOE band)
- Tail tone for weaker fits

**Avoid:** identical templates, truncated `...` mid-sentence, skills not in profile.

---

## Improvement 9: Docker reproduction (Stage 3)

```bash
docker build -t redrob-ranker .
docker run --rm -v "%cd%/data:/data" redrob-ranker \
  python rank.py --candidates /data/candidates.jsonl --out /data/submission.csv --no-embeddings
```

For full quality, mount precomputed `artifacts/` into the container.

---

## File map

| File | Responsibility |
|------|----------------|
| `precompute.py` | Offline embedding generation |
| `rank.py` | CLI entry point, writes CSV |
| `redrob_ranker/scorer.py` | Two-stage ranker |
| `redrob_ranker/features.py` | All merit + availability + tail features |
| `redrob_ranker/disqualifiers.py` | JD disqualifier multipliers |
| `redrob_ranker/honeypot.py` | Authenticity penalties |
| `redrob_ranker/embeddings.py` | Load/save dense vectors |
| `redrob_ranker/reasoning.py` | Per-row reasoning strings |
| `redrob_ranker/constants.py` | Weights, markers, thresholds |
| `analyze_gaps.py` | Local quality audit |

---

## Tuning guide (if you have 1 of 3 submissions left)

Edit `RankerConfig` in `scorer.py`:

| Knob | Direction | Effect |
|------|-----------|--------|
| `w_title` ↑ | More anti-keyword-stuffing | HR/Marketing with AI skills drop |
| `w_career_narrative` ↑ | More non-template fits rise | Search Engineers with IR in descriptions |
| `w_embedding` ↑ | More semantic, less lexical | Needs good precompute |
| `RECALL_POOL_SIZE` ↑ | Safer recall, slower | 8000 still fast |
| availability thresholds | Stricter in `availability_score()` | Low-response planted profiles drop in top 10 |

**Do not** manually edit `submission.csv` — Stage 3 reproduces from code.

---

## Submission checklist

- [ ] `python precompute.py` completed (optional but recommended)
- [ ] `python rank.py` finishes in <5 min on CPU
- [ ] `validate_submission.py` passes
- [ ] `analyze_gaps.py` — no research engineers in top 20, honeypot rate <10%
- [ ] Rename CSV to `team_xxx.csv`
- [ ] Fill `submission_metadata.yaml` (repo, sandbox, reproduce command)
- [ ] Push GitHub with real commit history
- [ ] Deploy `streamlit run app.py` for sandbox link
- [ ] Prepare 30-min architecture defense (Stages 3–5)

---

## Expected runtime

| Step | CPU time |
|------|----------|
| precompute.py (100K) | 10–20 min |
| rank.py without embeddings | ~60–90 sec |
| rank.py with embeddings | ~90–120 sec |

All within hackathon limits when precompute is separate from ranking.
