"""Recruiter-aligned feature extraction (v2)."""

from __future__ import annotations

import math
from datetime import date, datetime
from typing import Any

from .constants import (
    CAREER_IR_MARKERS,
    CORE_SKILLS,
    IDEAL_YOE_CENTER,
    IDEAL_YOE_SIGMA,
    MAX_PREFERRED_YOE,
    MIN_VIABLE_YOE,
    NEGATIVE_TITLE_KEYWORDS,
    POSITIVE_TITLE_KEYWORDS,
    PREFERRED_LOCATIONS,
    PRODUCT_SIGNAL_COMPANIES,
    STRONG_POSITIVE_TITLES,
    SUMMARY_IR_MARKERS,
    CONSULTING_FIRMS,
)
from .text_builder import build_candidate_document


def _gaussian_yoe(yoe: float) -> float:
    if yoe < MIN_VIABLE_YOE:
        return max(0.1, yoe / MIN_VIABLE_YOE * 0.5)
    if yoe > MAX_PREFERRED_YOE:
        return max(0.35, math.exp(-0.5 * ((yoe - MAX_PREFERRED_YOE) / 3.0) ** 2))
    z = (yoe - IDEAL_YOE_CENTER) / IDEAL_YOE_SIGMA
    return float(math.exp(-0.5 * z * z))


def title_alignment_score(candidate: dict[str, Any]) -> float:
    title = candidate["profile"].get("current_title", "").lower()
    headline = candidate["profile"].get("headline", "").lower()

    for neg in NEGATIVE_TITLE_KEYWORDS:
        if neg in title:
            return 0.05

    score = 0.2
    for strong in STRONG_POSITIVE_TITLES:
        if strong in title:
            score = max(score, 0.95)
    for pos in POSITIVE_TITLE_KEYWORDS:
        if pos in title or pos in headline:
            score = max(score, 0.75)

    if "software engineer" in title and ("ml" in headline or "machine learning" in headline):
        score = max(score, 0.65)

    history_titles = " ".join(h.get("title", "").lower() for h in candidate.get("career_history", []))
    if any(k in history_titles for k in ("search", "ranking", "retrieval", "nlp", "ml engineer")):
        score = min(1.0, score + 0.1)

    return score


def _ir_marker_score(text: str, markers: tuple[str, ...]) -> float:
    hits = sum(1 for m in markers if m in text)
    if hits >= 5:
        return 1.0
    if hits >= 3:
        return 0.75 + 0.05 * hits
    if hits >= 1:
        return 0.35 + 0.1 * hits
    return 0.0


def summary_narrative_score(candidate: dict[str, Any]) -> float:
    return _ir_marker_score(candidate["profile"].get("summary", "").lower(), SUMMARY_IR_MARKERS)


def career_narrative_score(candidate: dict[str, Any]) -> float:
    """IR/ranking evidence in career descriptions — catches non-template fits."""
    chunks = [
        h.get("description", "").lower()
        for h in candidate.get("career_history", [])
    ]
    chunks += [h.get("title", "").lower() for h in candidate.get("career_history", [])]
    return _ir_marker_score(" ".join(chunks), CAREER_IR_MARKERS)


def combined_narrative_score(candidate: dict[str, Any]) -> float:
    summary = summary_narrative_score(candidate)
    career = career_narrative_score(candidate)
    return max(summary, 0.55 * summary + 0.45 * career)


def skills_fit_score(candidate: dict[str, Any]) -> tuple[float, int, list[str]]:
    matched: list[str] = []
    weighted = 0.0
    max_possible = 0.0
    doc = build_candidate_document(candidate)

    for skill_name, weight in CORE_SKILLS.items():
        max_possible += weight
        if skill_name in doc:
            matched.append(skill_name)
            trust = 0.6
            for s in candidate.get("skills", []):
                if skill_name in s.get("name", "").lower():
                    prof_bonus = {
                        "beginner": 0.1,
                        "intermediate": 0.2,
                        "advanced": 0.35,
                        "expert": 0.4,
                    }.get(s.get("proficiency", ""), 0.15)
                    months = s.get("duration_months", 0)
                    endorse = s.get("endorsements", 0)
                    duration_trust = min(1.0, months / 18.0)
                    endorse_trust = min(1.0, endorse / 15.0)
                    trust = max(
                        trust,
                        0.35 + prof_bonus + 0.15 * duration_trust + 0.1 * endorse_trust,
                    )
            weighted += weight * min(1.0, trust)

    base = weighted / max_possible if max_possible else 0.0
    title_score = title_alignment_score(candidate)
    ai_skill_count = sum(
        1
        for s in candidate.get("skills", [])
        if any(
            k in s.get("name", "").lower()
            for k in ("llm", "rag", "nlp", "embedding", "vector", "pytorch", "tensorflow")
        )
    )
    if ai_skill_count >= 8 and title_score < 0.4:
        base *= 0.25
    return base, len(matched), matched


def career_quality_score(candidate: dict[str, Any]) -> float:
    profile = candidate["profile"]
    history = candidate.get("career_history", [])
    score = 0.5

    companies = [h.get("company", "").lower() for h in history]
    company_text = " ".join(companies)
    product_hits = sum(1 for c in PRODUCT_SIGNAL_COMPANIES if c in company_text)
    consulting_hits = sum(1 for c in CONSULTING_FIRMS if c in company_text)

    if product_hits:
        score += min(0.35, 0.1 * product_hits)
    if consulting_hits and product_hits == 0 and len(history) >= 2:
        score -= 0.25
    elif consulting_hits > product_hits:
        score -= 0.1

    desc_text = " ".join(h.get("description", "").lower() for h in history)
    production_markers = (
        "production", "shipped", "deployed", "p95", "latency",
        "a/b", "scale", "kubernetes", "serving",
    )
    marker_hits = sum(1 for m in production_markers if m in desc_text)
    score += min(0.25, 0.04 * marker_hits)

    if history:
        avg_tenure = sum(h.get("duration_months", 0) for h in history) / len(history)
        if avg_tenure < 18:
            score -= 0.15
        elif avg_tenure >= 30:
            score += 0.08

    current_company = profile.get("current_company", "").lower()
    if any(c in current_company for c in PRODUCT_SIGNAL_COMPANIES):
        score += 0.08

    return max(0.05, min(1.0, score))


def location_score(candidate: dict[str, Any]) -> float:
    loc = candidate["profile"].get("location", "").lower()
    country = candidate["profile"].get("country", "").lower()
    signals = candidate.get("redrob_signals", {})
    work_mode = signals.get("preferred_work_mode", "").lower()
    willing = signals.get("willing_to_relocate", False)

    if country and country not in ("india", "in"):
        base = 0.55 if willing else 0.35
    elif any(city in loc for city in PREFERRED_LOCATIONS):
        base = 1.0
    elif country in ("india", "in") or not country:
        base = 0.7
    else:
        base = 0.4

    # JD is hybrid — boost matching work mode preference
    if work_mode in ("hybrid", "onsite", "flexible"):
        base = min(1.0, base * 1.05)
    elif work_mode == "remote" and not willing:
        base *= 0.85

    return base


def availability_score(candidate: dict[str, Any], today: date | None = None) -> float:
    """
    Strong availability signal for top-rank ordering.
    JD: low response / inactive candidates are not hireable in practice.
    """
    today = today or date(2026, 6, 9)
    signals = candidate.get("redrob_signals", {})
    score = 1.0

    response = signals.get("recruiter_response_rate", 0)
    score *= 0.55 + 0.45 * min(1.0, response)

    if response < 0.2:
        score *= 0.72
    elif response < 0.35:
        score *= 0.88

    if not signals.get("open_to_work_flag", False):
        score *= 0.82

    last_active = signals.get("last_active_date")
    if last_active:
        try:
            days = (today - datetime.strptime(last_active[:10], "%Y-%m-%d").date()).days
            if days <= 30:
                score *= 1.05
            elif days > 365:
                score *= 0.65
            elif days > 180:
                score *= 0.88
        except ValueError:
            pass

    notice = signals.get("notice_period_days", 90)
    if notice <= 30:
        score *= 1.04
    elif notice > 90:
        score *= max(0.75, 1.0 - (notice - 90) / 240)

    interview_rate = signals.get("interview_completion_rate", 0.5)
    score *= 0.88 + 0.12 * interview_rate

    # Offer acceptance rate — engaged candidates accept offers
    offer_rate = signals.get("offer_acceptance_rate", -1)
    if offer_rate >= 0:
        score *= 0.85 + 0.15 * offer_rate

    # Response time — fast responders are more available
    avg_resp = signals.get("avg_response_time_hours", 24)
    if avg_resp <= 4:
        score *= 1.03
    elif avg_resp >= 72:
        score *= 0.92

    return max(0.35, min(1.15, score))


def tail_signals_score(candidate: dict[str, Any]) -> float:
    """Secondary signals to break ties in ranks 50-100."""
    signals = candidate.get("redrob_signals", {})
    score = 0.0

    saved = signals.get("saved_by_recruiters_30d", 0)
    score += min(0.25, saved * 0.04)

    views = signals.get("profile_views_received_30d", 0)
    score += min(0.15, views / 200)

    github = signals.get("github_activity_score", -1)
    if github >= 0:
        score += min(0.2, github / 100)

    assessments = signals.get("skill_assessment_scores", {})
    relevant = [
        v for k, v in assessments.items()
        if any(t in k.lower() for t in ("nlp", "ml", "python", "retrieval", "ranking", "embedding"))
    ]
    if relevant:
        score += min(0.25, sum(relevant) / (100 * len(relevant)))

    edu = candidate.get("education", [])
    if edu:
        tiers = [e.get("tier", "unknown") for e in edu]
        if "tier_1" in tiers:
            score += 0.12
        elif "tier_2" in tiers:
            score += 0.08
        fields = " ".join(e.get("field_of_study", "").lower() for e in edu)
        if any(f in fields for f in ("computer", "machine learning", "ai", "data science")):
            score += 0.08

    completeness = signals.get("profile_completeness_score", 0)
    score += min(0.1, completeness / 1000)

    return min(1.0, score)


def extract_features(candidate: dict[str, Any]) -> dict[str, Any]:
    profile = candidate["profile"]
    yoe = float(profile.get("years_of_experience", 0))
    title_s = title_alignment_score(candidate)
    summary_narr = summary_narrative_score(candidate)
    career_narr = career_narrative_score(candidate)
    narrative_s = combined_narrative_score(candidate)
    skills_s, skill_count, skill_names = skills_fit_score(candidate)
    career_s = career_quality_score(candidate)
    loc_s = location_score(candidate)
    yoe_s = _gaussian_yoe(yoe)
    avail_s = availability_score(candidate)
    tail_s = tail_signals_score(candidate)

    history = candidate.get("career_history", [])
    employers = [h.get("company", "") for h in history[:3] if h.get("company")]

    return {
        "candidate_id": candidate["candidate_id"],
        "title": profile.get("current_title", ""),
        "current_company": profile.get("current_company", ""),
        "employers": employers,
        "yoe": yoe,
        "location": profile.get("location", ""),
        "country": profile.get("country", ""),
        "title_score": title_s,
        "summary_narrative_score": summary_narr,
        "career_narrative_score": career_narr,
        "narrative_score": narrative_s,
        "skills_score": skills_s,
        "skill_count": skill_count,
        "skill_names": skill_names,
        "career_score": career_s,
        "location_score": loc_s,
        "yoe_score": yoe_s,
        "availability_score": avail_s,
        "tail_score": tail_s,
        "response_rate": candidate.get("redrob_signals", {}).get("recruiter_response_rate", 0),
        "open_to_work": candidate.get("redrob_signals", {}).get("open_to_work_flag", False),
        "notice_period": candidate.get("redrob_signals", {}).get("notice_period_days", 0),
        "saved_by_recruiters": candidate.get("redrob_signals", {}).get("saved_by_recruiters_30d", 0),
    }
