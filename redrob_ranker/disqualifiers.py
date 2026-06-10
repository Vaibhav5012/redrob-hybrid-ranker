"""JD explicit disqualifier detection."""

from __future__ import annotations

from typing import Any

from .constants import (
    CONSULTING_FIRMS,
    CV_SPEECH_SKILLS,
    IR_NLP_SKILLS,
    LANGCHAIN_ONLY_MARKERS,
    NON_TECH_ROLE_MARKERS,
    RESEARCH_TITLE_KEYWORDS,
)


def _skill_names(candidate: dict[str, Any]) -> list[str]:
    return [s.get("name", "").lower() for s in candidate.get("skills", [])]


def research_penalty(candidate: dict[str, Any]) -> float:
    """JD: no pure research without production deployment."""
    title = candidate["profile"].get("current_title", "").lower()
    headline = candidate["profile"].get("headline", "").lower()
    if any(k in title or k in headline for k in RESEARCH_TITLE_KEYWORDS):
        history_text = " ".join(
            h.get("description", "").lower() for h in candidate.get("career_history", [])
        )
        production = any(
            m in history_text
            for m in ("production", "deployed", "shipped", "kubernetes", "serving", "p95")
        )
        return 0.45 if production else 0.25
    return 1.0


def consulting_only_penalty(candidate: dict[str, Any]) -> float:
    """JD: consulting-only careers are a poor fit."""
    history = candidate.get("career_history", [])
    if len(history) < 2:
        return 1.0

    companies = [h.get("company", "").lower() for h in history]
    consulting_roles = sum(
        1 for c in companies if any(firm in c for firm in CONSULTING_FIRMS)
    )
    product_roles = sum(
        1
        for h in history
        if not any(f in h.get("company", "").lower() for f in CONSULTING_FIRMS)
        and h.get("company", "").lower() not in ("", "unknown")
    )
    if consulting_roles >= 2 and product_roles == 0:
        return 0.5
    if consulting_roles > product_roles + 1:
        return 0.75
    return 1.0


def cv_speech_without_ir_penalty(candidate: dict[str, Any]) -> float:
    """JD: CV/speech expertise without NLP/IR exposure."""
    skills = _skill_names(candidate)
    cv_hits = sum(1 for s in skills if any(c in s for c in CV_SPEECH_SKILLS))
    ir_hits = sum(1 for s in skills if any(c in s for c in IR_NLP_SKILLS))
    if cv_hits >= 3 and ir_hits <= 1:
        return 0.4
    if cv_hits >= 2 and ir_hits == 0:
        return 0.55
    return 1.0


def langchain_tourist_penalty(candidate: dict[str, Any]) -> float:
    """JD: recent LangChain-only AI experience without depth."""
    skills = candidate.get("skills", [])
    yoe = float(candidate["profile"].get("years_of_experience", 0))
    langchain_months = 0
    deep_ml_months = 0
    for s in skills:
        name = s.get("name", "").lower()
        months = s.get("duration_months", 0)
        if any(m in name for m in LANGCHAIN_ONLY_MARKERS):
            langchain_months += months
        if any(
            m in name
            for m in ("pytorch", "tensorflow", "retrieval", "ranking", "embedding", "ndcg")
        ):
            deep_ml_months += months
    if langchain_months > 0 and deep_ml_months < 12 and yoe < 6:
        return 0.55
    return 1.0


def career_consistency_penalty(candidate: dict[str, Any]) -> float:
    """Penalize AI/ML titles backed by mostly non-technical career descriptions."""
    title = candidate["profile"].get("current_title", "").lower()
    if not any(k in title for k in ("engineer", "scientist", "ml", "ai", "nlp", "data")):
        return 1.0

    descs = [h.get("description", "").lower() for h in candidate.get("career_history", [])]
    if not descs:
        return 1.0

    non_tech = sum(
        1 for d in descs if any(marker in d for marker in NON_TECH_ROLE_MARKERS)
    )
    if non_tech >= 3:
        return 0.2
    if non_tech >= 2:
        return 0.45
    if non_tech == 1:
        return 0.8
    return 1.0


def disqualifier_multiplier(candidate: dict[str, Any]) -> float:
    """Combined JD disqualifier multiplier in roughly [0.1, 1.0]."""
    penalties = [
        research_penalty(candidate),
        consulting_only_penalty(candidate),
        cv_speech_without_ir_penalty(candidate),
        langchain_tourist_penalty(candidate),
        career_consistency_penalty(candidate),
    ]
    combined = 1.0
    for p in penalties:
        combined *= p
    return max(0.08, combined)
