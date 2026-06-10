"""Analyze ranking gaps for hackathon improvement."""
import csv
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))

from redrob_ranker.honeypot import honeypot_score
from redrob_ranker.loader import load_candidates, load_job_description
from redrob_ranker.scorer import HybridRanker

BASE = pathlib.Path(__file__).parent
CAND_PATH = (
    BASE
    / "[PUB] India_runs_data_and_ai_challenge"
    / "[PUB] India_runs_data_and_ai_challenge"
    / "India_runs_data_and_ai_challenge"
    / "candidates.jsonl"
)


def main():
    sub = {
        r["candidate_id"]: int(r["rank"])
        for r in csv.DictReader(open(BASE / "submission.csv", encoding="utf-8"))
    }
    cands = {}
    with CAND_PATH.open(encoding="utf-8") as f:
        for line in f:
            c = json.loads(line)
            cands[c["candidate_id"]] = c

    print("=== TOP 20 - potential ground-truth risks ===")
    for rank, cid in sorted((v, k) for k, v in sub.items())[:20]:
        c = cands[cid]
        p = c["profile"]
        issues = []
        yoe = p["years_of_experience"]
        if yoe > 12:
            issues.append(f"high yoe {yoe}")
        if yoe < 4:
            issues.append(f"low yoe {yoe}")
        if p["country"].lower() not in ("india", "in", ""):
            issues.append(f"country={p['country']}")
        expert0 = sum(
            1
            for s in c["skills"]
            if s.get("proficiency") == "expert" and s.get("duration_months", 0) == 0
        )
        if expert0:
            issues.append(f"expert@0mo={expert0}")
        tm = sum(h["duration_months"] for h in c["career_history"])
        if tm > yoe * 12 + 36:
            issues.append("timeline overlap")
        rr = c["redrob_signals"]["recruiter_response_rate"]
        if rr < 0.3:
            issues.append(f"low response {rr:.2f}")
        notice = c["redrob_signals"]["notice_period_days"]
        if notice > 90:
            issues.append(f"notice {notice}d")
        auth = honeypot_score(c)
        if auth < 0.7:
            issues.append(f"auth={auth:.2f}")
        print(rank, cid, p["current_title"][:35], "|", "; ".join(issues) or "ok")

    jd = load_job_description()
    all_cands = load_candidates(CAND_PATH)
    ranker = HybridRanker(jd)
    ranker.fit(all_cands)
    scored = ranker.score_all()
    scored.sort(key=lambda x: (-x[1], x[0]))

    print("\n=== Ranks 90-110 (cutoff zone) ===")
    for i, (cid, score, feats) in enumerate(scored[89:110], start=90):
        print(
            i,
            cid,
            f"{score:.4f}",
            feats["title"][:28],
            f"yoe={feats['yoe']:.1f}",
            f"auth={feats['authenticity']:.2f}",
        )

    # Honeypots in top 50 by our rank
    print("\n=== Low authenticity in top 50 ===")
    for rank, cid in sorted((v, k) for k, v in sub.items())[:50]:
        c = cands[cid]
        auth = honeypot_score(c)
        if auth < 0.75:
            print(rank, cid, auth, c["profile"]["current_title"])

    # Candidates with search/ranking in career but not planted template
    print("\n=== Non-template but strong IR career (score top 200) ===")
    count = 0
    for cid, score, feats in scored[:200]:
        c = cands[cid]
        if feats["narrative_score"] >= 0.9:
            continue
        hist = " ".join(h.get("description", "").lower() for h in c["career_history"])
        if any(
            m in hist
            for m in ("hybrid retrieval", "learning-to-rank", "ndcg", "vector search", "bm25")
        ):
            print(
                sub.get(cid, "101+"),
                cid,
                f"{score:.4f}",
                feats["title"],
                feats["yoe"],
            )
            count += 1
            if count >= 15:
                break


if __name__ == "__main__":
    main()
