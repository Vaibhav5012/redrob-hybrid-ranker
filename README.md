---
title: Redrob AI Recruiter
emoji: 🎯
colorFrom: indigo
colorTo: purple
sdk: streamlit
sdk_version: "1.45.0"
app_file: app.py
pinned: false
license: apache-2.0
---

# Redrob AI Recruiter v2

Two-stage hybrid ranker for the **Redrob Intelligent Candidate Discovery & Ranking Challenge**. Ranks ~100K profiles against a Senior AI Engineer JD using a recall → rerank pipeline with behavioral scoring — not keyword stuffing.

## Architecture

```
┌─────────────────── OFFLINE (once, ~6 min) ───────────────────┐
│  precompute.py                                                │
│    → Download sentence-transformers/all-MiniLM-L6-v2          │
│    → Encode 100K candidate documents → 384-dim vectors        │
│    → Save artifacts/*.npy + candidate_ids.json                │
└───────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────── ONLINE (<3 min, CPU, no network) ─────────┐
│  rank.py                                                      │
│                                                               │
│  STAGE 1 — Recall (~8-10K candidates)                         │
│    BM25 top 5K ∪ Dense embedding top 5K ∪ Strong titles       │
│                                                               │
│  STAGE 2 — Rerank                                             │
│    merit = Σ(feature × weight) for recalled pool only         │
│    composite = merit × availability × disqualifier × auth     │
│    Sort by composite → top 100 → submission.csv               │
└───────────────────────────────────────────────────────────────┘
```

### Merit Features (weighted sum)

| Feature | Weight | Source |
|---------|--------|--------|
| Title alignment | 0.24 | Current title + career history titles |
| Semantic score | 0.14 | 0.65×(BM25+TF-IDF) + 0.35×dense cosine |
| Skills fit | 0.17 | 26 core skills with trust scoring (proficiency, duration, endorsements) |
| Summary narrative | 0.13 | IR/ranking markers in profile summary |
| Career narrative | 0.10 | IR/ranking markers in career descriptions |
| Career quality | 0.12 | Product vs consulting, production markers, tenure |
| YOE fit | 0.04 | Gaussian centered at 6.5y, σ=2.0 |
| Location | 0.03 | India + Pune/Noida preferred + work mode |
| Tail signals | 0.03 | GitHub, assessments, education, recruiter saves |

### Multiplicative Factors

| Factor | What it checks |
|--------|---------------|
| **Availability** | Response rate, open-to-work, last active, notice period, offer acceptance rate |
| **Disqualifiers** | Research-only, consulting-only, CV/speech-without-IR, LangChain tourist, career inconsistency |
| **Authenticity** | Expert@0mo, timeline overflow, date inconsistencies, buzz stuffing, unverified identity |

## Quick Start

```bash
pip install -r requirements.txt

# Step 1: Offline embedding precomputation (recommended, ~6 min)
python precompute.py --candidates ./candidates.jsonl --jd ./job_description.txt --artifacts ./artifacts

# Step 2: Rank candidates (<3 min, CPU only, no network)
python rank.py --candidates ./candidates.jsonl --out ./submission.csv --artifacts ./artifacts

# Without embeddings (still works, slightly weaker):
python rank.py --candidates ./candidates.jsonl --out ./submission.csv --no-embeddings
```

### Validate & Audit

```bash
# Format validation
python validate_submission.py submission.csv

# Quality audit (top-20 risks, cutoff zone, honeypots, career-IR candidates)
python analyze_gaps.py
```

## Streamlit Sandbox

```bash
streamlit run app.py
```

Upload `sample_candidates.json` or a JSONL slice (≤500 for demo). Also deployed on [HuggingFace Spaces](https://huggingface.co/spaces/).

## Docker Reproduction

```bash
docker build -t redrob-ranker .
docker run -v ./candidates.jsonl:/app/candidates.jsonl redrob-ranker
```

## Compute Constraints

- **CPU only**, no GPU, no network during ranking
- Completes in **~2 min** on 100K candidates (16 GB RAM)
- Precompute is offline-only, one-time (~6 min)
- No hosted LLM APIs — all scoring is deterministic

## Project Layout

```
rank.py                       # CLI entry point (online ranking)
precompute.py                 # Offline embedding generation
app.py                        # Streamlit demo for HuggingFace Spaces
analyze_gaps.py               # Local quality audit
redrob_ranker/
  scorer.py                   # Two-stage ranker (recall → rerank)
  features.py                 # Title, skills, career, behavioral, tail features
  disqualifiers.py            # JD-specific disqualifier penalties
  honeypot.py                 # Profile authenticity detection
  reasoning.py                # Per-candidate reasoning strings
  embeddings.py               # Dense embedding load/similarity
  constants.py                # Weights, markers, thresholds
  text_builder.py             # Candidate document construction for BM25/TF-IDF
  loader.py                   # Data loading utilities
job_description.txt           # Target JD
submission.csv                # Generated top-100 ranking
artifacts/                    # Precomputed embeddings (optional)
  candidate_embeddings.npy    # 100K × 384 float32 vectors
  jd_embedding.npy            # JD query vector
  candidate_ids.json          # ID order mapping
Dockerfile                    # Stage 3 reproduction
submission_metadata.yaml      # Required hackathon metadata
```

## Design Decisions & Trade-offs

1. **Two-stage recall → rerank** over single-pass scoring: enables expensive feature extraction on only ~8K candidates instead of 100K, keeping runtime under 3 min.

2. **Multiplicative behavioral factors** (availability × disqualifier × authenticity) rather than additive: ensures a single strong negative signal (e.g., honeypot profile) can dramatically reduce rank regardless of merit, matching how real recruiters operate.

3. **Career narrative scoring** separate from summary narrative: catches Search Engineers and Recommendation Systems Engineers whose career descriptions contain IR/ranking evidence even when their summary doesn't use template phrases.

4. **Anti-keyword-stuffing gates**: if a candidate has 8+ AI buzzword skills but a non-technical title (score < 0.4), their skills score is penalized 75% — this targets the intentional trap in the dataset.

5. **Dense embeddings as semantic boost** (not replacement): lexical scoring (BM25 + TF-IDF) remains the base, with dense cosine similarity weighted at 35% of the semantic score. This prevents embedding failures from overriding strong lexical matches.
