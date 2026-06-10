FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY redrob_ranker/ redrob_ranker/
COPY rank.py precompute.py job_description.txt ./

# Copy precomputed artifacts if available (optional — ranking works without them)
COPY artifacts/ artifacts/

# Default: full ranking with embeddings if artifacts present
ENTRYPOINT ["python", "rank.py"]
CMD ["--candidates", "./candidates.jsonl", "--out", "./submission.csv", "--artifacts", "./artifacts"]
