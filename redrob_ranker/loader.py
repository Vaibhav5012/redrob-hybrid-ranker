"""Load candidates and job description."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

from .constants import DEFAULT_JD_PATH


def load_job_description(path: Path | str | None = None) -> str:
    jd_path = Path(path) if path else DEFAULT_JD_PATH
    return jd_path.read_text(encoding="utf-8")


def iter_candidates(path: Path | str) -> Iterator[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_candidates(path: Path | str) -> list[dict[str, Any]]:
    return list(iter_candidates(path))
