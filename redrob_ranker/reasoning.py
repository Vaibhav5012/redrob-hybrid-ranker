"""Generate unique, recruiter-style reasoning per candidate."""

from __future__ import annotations

import hashlib
from typing import Any


# Multiple JD-specific hooks to avoid templated phrasing
_JD_HOOKS_NARRATIVE = [
    "aligns with JD need for production retrieval/ranking and eval frameworks",
    "matches requirement for embeddings-based search at scale with NDCG/MRR evaluation",
    "fits JD requirement for hybrid search infra and offline-to-online evaluation loops",
    "demonstrates production retrieval experience the JD explicitly prioritizes",
    "career evidence maps to JD's core need: shipped ranking/recommendation systems",
]

_JD_HOOKS_SKILLS = [
    "covers JD core stack (embeddings, vector DBs, retrieval, ranking)",
    "skill set spans the JD-required toolchain: sentence-transformers, vector search, Python",
    "technical depth in retrieval and ranking aligns with JD's primary requirements",
]

_JD_HOOKS_TITLE = [
    "title matches Senior AI Engineer scope",
    "role scope aligns with founding-team AI engineering position",
    "current title directly relevant to the Senior AI Engineer opening",
]


def _select_hook(hooks: list[str], candidate_id: str) -> str:
    """Deterministic but varied selection based on candidate_id hash."""
    idx = int(hashlib.md5(candidate_id.encode()).hexdigest(), 16) % len(hooks)
    return hooks[idx]


def _rank_tone(merit: float, rank_hint: int | None = None) -> str | None:
    """Add rank-aware context for tail candidates."""
    if merit >= 0.7:
        return None  # top candidates speak for themselves
    if merit >= 0.5:
        return "solid fit with minor gaps in JD-ideal profile"
    if merit >= 0.4:
        return "adjacent fit — included for IR-relevant skills and strong engagement signals"
    return "borderline fit — ranked on transferable skills and behavioral indicators"


def _jd_hook(feats: dict[str, Any]) -> str | None:
    cid = feats.get("candidate_id", "")
    if feats.get("career_narrative_score", 0) >= 0.6 or feats.get("summary_narrative_score", 0) >= 0.6:
        return _select_hook(_JD_HOOKS_NARRATIVE, cid)
    if feats.get("skills_score", 0) >= 0.5:
        return _select_hook(_JD_HOOKS_SKILLS, cid)
    if feats.get("title_score", 0) >= 0.75:
        return _select_hook(_JD_HOOKS_TITLE, cid)
    return None


def build_reasoning(feats: dict[str, Any]) -> str:
    """Rich recruiter-style reasoning: specific facts, JD link, honest concerns."""
    parts: list[str] = []

    # 1. Identity line
    title = feats.get("title", "Professional")
    company = feats.get("current_company", "")
    yoe = feats.get("yoe", 0)
    if company:
        parts.append(f"{title} at {company} ({yoe:.1f} yrs)")
    else:
        parts.append(f"{title} with {yoe:.1f} yrs experience")

    # 2. Career trajectory
    employers = feats.get("employers", [])
    if employers:
        parts.append(f"previously at {', '.join(employers[:2])}")

    # 3. Key skills (up to 4 for more specificity)
    skill_names = feats.get("skill_names", [])
    if skill_names:
        parts.append(f"strong on {', '.join(skill_names[:4])}")

    # 4. JD connection (varied phrasing)
    jd = _jd_hook(feats)
    if jd:
        parts.append(jd)

    # 5. Location
    loc = feats.get("location", "")
    if loc and feats.get("location_score", 0) >= 0.9:
        parts.append(f"located in {loc} (JD-preferred geography)")

    # 6. Positive signals
    rr = feats.get("response_rate", 0)
    if rr >= 0.7:
        parts.append(f"high recruiter engagement ({rr:.0%} response rate)")
    elif rr < 0.25 and rr > 0:
        parts.append(f"concern: low recruiter response rate ({rr:.0%}) may indicate passive status")

    if feats.get("open_to_work"):
        parts.append("open to work")
    elif feats.get("merit_score", 0) > 0.6:
        parts.append("not flagged open-to-work")

    # 7. Honest concerns (specific, not generic)
    notice = feats.get("notice_period", 0)
    if notice > 90:
        parts.append(f"notice period {notice} days may delay start beyond JD's preferred timeline")

    if feats.get("yoe", 0) > 12:
        band_diff = feats["yoe"] - 8
        parts.append(f"{feats['yoe']:.1f} yrs experience is {band_diff:.0f}+ yrs above JD's ideal 5-8 yr band")

    country = feats.get("country", "").lower()
    if country and country not in ("india", "in", ""):
        parts.append(f"based in {feats.get('country')} — relocation case-by-case per JD")

    if feats.get("disqualifier_multiplier", 1.0) < 0.6:
        parts.append("JD disqualifier signals (research-only/consulting-only/domain-mismatch) noted")

    if feats.get("authenticity", 1.0) < 0.65:
        parts.append("profile consistency flags reduce ranking confidence")

    # 8. Rank-aware tone
    rank_tone = _rank_tone(feats.get("merit_score", 0))
    if rank_tone:
        parts.append(rank_tone)

    # Join with ". " — allow up to 500 chars for richer reasoning
    sentence = ". ".join(parts[:9])
    if len(sentence) > 500:
        sentence = sentence[:497] + "..."
    return sentence
