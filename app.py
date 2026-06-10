"""Streamlit sandbox for Redrob AI Recruiter (small-sample demo)."""

from __future__ import annotations

import json
from pathlib import Path
import pandas as pd
import streamlit as st

from redrob_ranker.loader import load_job_description
from redrob_ranker.reasoning import build_reasoning
from redrob_ranker.scorer import HybridRanker

st.set_page_config(page_title="Redrob AI Recruiter", layout="wide", page_icon="🎯")

# Premium CSS styling
st.markdown("""
<style>
    /* Gradient Header */
    .stApp > header {
        background-color: transparent;
    }
    .header-box {
        background: linear-gradient(90deg, #4F46E5 0%, #7C3AED 100%);
        padding: 2rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 2rem;
    }
    .header-box h1 {
        color: white !important;
        margin: 0;
        font-size: 2.5rem;
        font-weight: 800;
    }
    .header-box p {
        font-size: 1.1rem;
        opacity: 0.9;
        margin-top: 0.5rem;
    }
    
    /* Metrics / Badges */
    .metric-card {
        background-color: #1E1E2E;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        border: 1px solid #333;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #A78BFA;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #9CA3AF;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .badge-green { background-color: #065F46; color: #34D399; padding: 0.2rem 0.6rem; border-radius: 12px; font-size: 0.8rem; font-weight: bold; }
    .badge-yellow { background-color: #78350F; color: #FBBF24; padding: 0.2rem 0.6rem; border-radius: 12px; font-size: 0.8rem; font-weight: bold; }
    .badge-red { background-color: #7F1D1D; color: #F87171; padding: 0.2rem 0.6rem; border-radius: 12px; font-size: 0.8rem; font-weight: bold; }

    /* Expandable grid */
    .feature-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
        gap: 1rem;
        margin-bottom: 1rem;
    }
    .feature-box {
        background-color: rgba(255, 255, 255, 0.05);
        padding: 0.8rem;
        border-radius: 6px;
    }
    .feature-box strong {
        display: block;
        color: #9CA3AF;
        font-size: 0.8rem;
        margin-bottom: 0.3rem;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="header-box">
    <h1>🎯 Redrob AI Recruiter</h1>
    <p>Hybrid BM25 + TF-IDF + Recruiter-Feature Ranking (CPU, offline)</p>
</div>
""", unsafe_allow_html=True)

# Architecture Expander
with st.expander("🛠️ Architecture & Pipeline View"):
    st.markdown("""
    ### 1. Offline Precompute (100K Candidates)
    - Download **sentence-transformers/all-MiniLM-L6-v2**
    - Encode all candidate documents into 384-dimensional dense vectors
    - Save to `.npy` artifacts for fast retrieval

    ### 2. Online Ranking (<3 mins)
    **Stage 1: Recall (Top ~8-10K)**
    - BM25 Lexical Top 5K
    - Dense Embedding Top 5K
    - Union with strong-title candidates

    **Stage 2: Rerank & Score**
    - **Merit Score:** Title Alignment (0.24) + Semantic Relevance (0.14) + Skills Fit (0.17) + Career Narrative (0.10) + Summary Narrative (0.13) + Career Quality (0.12) + YOE (0.04) + Location (0.03) + Tail (0.03)
    - **Multipliers:** Availability × Disqualifiers × Authenticity
    """)

# Sidebar
st.sidebar.title("Configuration")
jd_path = Path(__file__).parent / "job_description.txt"
jd_text = load_job_description(jd_path)

with st.sidebar.expander("📄 Job Description (Senior AI Engineer)", expanded=False):
    st.text(jd_text[:1500] + "...\n\n[Truncated for display]")

uploaded = st.sidebar.file_uploader(
    "Upload candidate JSONL (≤500 for demo)",
    type=["json", "jsonl"],
)

sample_path = (
    Path(__file__).parent
    / "[PUB] India_runs_data_and_ai_challenge"
    / "[PUB] India_runs_data_and_ai_challenge"
    / "India_runs_data_and_ai_challenge"
    / "sample_candidates.json"
)

use_sample = st.sidebar.checkbox("Use bundled sample dataset", value=uploaded is None)
top_n = st.sidebar.slider("Top N to display", min_value=10, max_value=100, value=25, step=5)
run_btn = st.sidebar.button("🚀 Rank Candidates", type="primary", use_container_width=True)

# Main logic
if run_btn:
    candidates = []
    if use_sample and sample_path.exists():
        candidates = json.loads(sample_path.read_text(encoding="utf-8"))
    elif uploaded:
        raw = uploaded.read().decode("utf-8")
        if uploaded.name.endswith(".jsonl"):
            candidates = [json.loads(line) for line in raw.splitlines() if line.strip()]
        else:
            candidates = json.loads(raw)
    else:
        st.error("Upload a file or enable the sample dataset.")
        st.stop()

    if len(candidates) > 500:
        st.warning(f"Demo capped at 500 candidates. Using first 500.")
        candidates = candidates[:500]

    with st.spinner("Analyzing profiles and computing hybrid scores..."):
        ranker = HybridRanker(jd_text)
        ranker.fit(candidates)
        scored = ranker.score_all()
        scored.sort(key=lambda x: (-x[1], x[0]))
        top = scored[:top_n]

    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""<div class="metric-card"><div class="metric-value">{len(candidates)}</div><div class="metric-label">Profiles</div></div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""<div class="metric-card"><div class="metric-value">{len(scored)}</div><div class="metric-label">Recalled</div></div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""<div class="metric-card"><div class="metric-value">{top[0][1]:.2f}</div><div class="metric-label">Top Score</div></div>""", unsafe_allow_html=True)
    with col4:
        avg_score = sum(s[1] for s in top) / len(top)
        st.markdown(f"""<div class="metric-card"><div class="metric-value">{avg_score:.2f}</div><div class="metric-label">Avg Top-{top_n} Score</div></div>""", unsafe_allow_html=True)

    st.markdown("### Score Distribution (Top N)")
    scores_df = pd.DataFrame([s[1] for s in top], columns=["Composite Score"])
    st.bar_chart(scores_df)

    st.markdown(f"### Top {top_n} Ranked Candidates")
    
    # Export prep
    export_rows = []

    for rank, (cid, score, feats) in enumerate(top, start=1):
        reasoning = build_reasoning(feats)
        
        # Color badge for score
        badge_class = "badge-green" if score > 0.7 else ("badge-yellow" if score > 0.4 else "badge-red")
        
        export_rows.append({
            "candidate_id": cid,
            "rank": rank,
            "score": round(score, 4),
            "reasoning": reasoning
        })

        with st.expander(f"#{rank} | {feats['title']} ({feats['yoe']:.1f}y) | Score: {score:.3f}"):
            st.markdown(f"""
            <div style="margin-bottom: 1rem;">
                <span class="{badge_class}">Score: {score:.3f}</span>
                <span style="margin-left: 0.5rem; opacity: 0.8; font-size: 0.9em;">ID: {cid}</span>
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f"**Reasoning:** _{reasoning}_")
            
            # Sub-scores
            st.markdown("**Score Components Breakdown**")
            breakdown_data = pd.DataFrame({
                "Score": [
                    feats.get('title_score', 0),
                    feats.get('skills_score', 0),
                    feats.get('career_score', 0),
                    feats.get('semantic_score', 0),
                    feats.get('narrative_score', 0)
                ]
            }, index=["Title", "Skills", "Career", "Semantic", "Narrative"])
            st.bar_chart(breakdown_data, height=200)
            
            # Grid of multipliers
            st.markdown("""
            <div class="feature-grid">
                <div class="feature-box"><strong>Merit (Base)</strong>{merit:.3f}</div>
                <div class="feature-box"><strong>Availability Mult</strong>{avail:.3f}</div>
                <div class="feature-box"><strong>Disqualifier Mult</strong>{disq:.3f}</div>
                <div class="feature-box"><strong>Authenticity (Honeypot)</strong>{auth:.3f}</div>
                <div class="feature-box"><strong>Composite (Final)</strong>{comp:.3f}</div>
            </div>
            """.format(
                merit=feats.get('merit_score', 0),
                avail=feats.get('availability_multiplier', 1),
                disq=feats.get('disqualifier_multiplier', 1),
                auth=feats.get('authenticity', 1),
                comp=score
            ), unsafe_allow_html=True)

    # Download
    csv_content = "candidate_id,rank,score,reasoning\\n" + "\\n".join(
        f'{r["candidate_id"]},{r["rank"]},{r["score"]},"{r["reasoning"]}"'
        for r in export_rows
    )
    st.download_button("📥 Download CSV Submission", csv_content, file_name="ranked_sample.csv", mime="text/csv", use_container_width=True)

st.markdown("---")
st.markdown("""
<div style="opacity: 0.7; font-size: 0.9em; text-align: center;">
    <strong>Methodology:</strong> Two-stage hybrid ranker utilizing BM25 + dense embedding recall (MiniLM-L6-v2), followed by a deterministic feature fusion reranker. 
    Scores candidates on merit (title, skills, semantic fit, career trajectory) and applies multipliers for behavioral availability and JD disqualifiers. 
    Authenticity checks penalize honeypot profiles automatically.
</div>
""", unsafe_allow_html=True)
