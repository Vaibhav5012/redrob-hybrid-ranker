"""Detect honeypot / impossible profiles per challenge spec."""

from __future__ import annotations

from datetime import date, datetime
from typing import Any


def _parse_date(value: str | None) -> date | None:
    if not value:
        return None
    try:
        return datetime.strptime(value[:10], "%Y-%m-%d").date()
    except ValueError:
        return None


def honeypot_score(candidate: dict[str, Any], today: date | None = None) -> float:
    """
    Return authenticity multiplier in [0, 1].
    1.0 = clean profile, near 0 = likely honeypot.
    """
    today = today or date(2026, 6, 9)
    penalty = 0.0

    profile = candidate["profile"]
    yoe = float(profile.get("years_of_experience", 0))
    skills = candidate.get("skills", [])
    history = candidate.get("career_history", [])

    expert_zero = sum(
        1
        for s in skills
        if s.get("proficiency") == "expert" and s.get("duration_months", 0) == 0
    )
    if expert_zero >= 3:
        penalty += 0.35 + 0.05 * (expert_zero - 3)

    advanced_count = sum(
        1 for s in skills if s.get("proficiency") in ("advanced", "expert")
    )
    if advanced_count >= 10 and yoe < 3:
        penalty += 0.4

    total_months = sum(int(h.get("duration_months", 0)) for h in history)
    expected_months = yoe * 12
    if total_months > expected_months + 36:
        penalty += min(0.5, (total_months - expected_months - 36) / 120)

    if total_months < max(12, expected_months * 0.4) and yoe >= 5:
        penalty += 0.25

    for role in history:
        start = _parse_date(role.get("start_date"))
        end = _parse_date(role.get("end_date"))
        months = int(role.get("duration_months", 0))
        if start and end and months > 0:
            span = (end.year - start.year) * 12 + (end.month - start.month)
            if months > span + 6:
                penalty += 0.15

    signals = candidate.get("redrob_signals", {})
    last_active = _parse_date(signals.get("last_active_date"))
    if last_active:
        inactive_days = (today - last_active).days
        if inactive_days > 365:
            penalty += 0.05  # mild — behavioral, not honeypot

    # Suspicious: beginner proficiency claimed for many AI buzzwords with high endorsements
    buzz = ("rag", "llm", "langchain", "pinecone", "vector", "embedding")
    stuffed = 0
    for s in skills:
        name = s.get("name", "").lower()
        if any(b in name for b in buzz):
            if s.get("proficiency") in ("advanced", "expert") and s.get(
                "duration_months", 0
            ) < 3:
                stuffed += 1
    if stuffed >= 4:
        penalty += 0.2

    # Unverified identity — legitimate candidates tend to have verified profiles
    verified_count = sum([
        bool(signals.get("verified_email", False)),
        bool(signals.get("verified_phone", False)),
        bool(signals.get("linkedin_connected", False)),
    ])
    if verified_count == 0:
        penalty += 0.15
    elif verified_count == 1:
        penalty += 0.05

    penalty = min(0.95, penalty)
    return max(0.05, 1.0 - penalty)
