"""Build searchable text representations for candidates."""

from __future__ import annotations

from typing import Any


def _skill_line(skill: dict[str, Any]) -> str:
    name = skill.get("name", "")
    prof = skill.get("proficiency", "")
    months = skill.get("duration_months", 0)
    endorsements = skill.get("endorsements", 0)
    return f"{name} ({prof}, {months}mo, {endorsements} endorsements)"


def build_candidate_document(candidate: dict[str, Any]) -> str:
    """Flatten a candidate profile into a single searchable document."""
    profile = candidate["profile"]
    parts = [
        profile.get("headline", ""),
        profile.get("summary", ""),
        profile.get("current_title", ""),
        profile.get("current_company", ""),
        profile.get("current_industry", ""),
        profile.get("location", ""),
        profile.get("country", ""),
    ]

    for role in candidate.get("career_history", []):
        parts.extend(
            [
                role.get("title", ""),
                role.get("company", ""),
                role.get("industry", ""),
                role.get("description", ""),
            ]
        )

    for edu in candidate.get("education", []):
        parts.extend(
            [
                edu.get("degree", ""),
                edu.get("field_of_study", ""),
                edu.get("institution", ""),
            ]
        )

    for skill in candidate.get("skills", []):
        parts.append(_skill_line(skill))

    for cert in candidate.get("certifications", []):
        parts.append(f"{cert.get('name', '')} {cert.get('issuer', '')}")

    signals = candidate.get("redrob_signals", {})
    assessments = signals.get("skill_assessment_scores", {})
    if assessments:
        parts.append(
            "assessments: "
            + ", ".join(f"{k}={v}" for k, v in sorted(assessments.items()))
        )

    return " ".join(p for p in parts if p).lower()


def tokenize(text: str) -> list[str]:
    """Simple whitespace + punctuation tokenizer for BM25."""
    import re

    return re.findall(r"[a-z0-9+#./-]+", text.lower())
