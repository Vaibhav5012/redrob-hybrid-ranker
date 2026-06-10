"""Job requirements and scoring constants derived from the released JD."""

from pathlib import Path

# --- Title signals ---

STRONG_POSITIVE_TITLES = (
    "senior ai engineer",
    "lead ai engineer",
    "staff machine learning engineer",
    "staff ml engineer",
    "senior nlp engineer",
    "senior machine learning engineer",
    "senior applied scientist",
    "applied scientist",
    "search engineer",
    "recommendation systems engineer",
)

POSITIVE_TITLE_KEYWORDS = (
    "ai engineer",
    "ml engineer",
    "machine learning engineer",
    "nlp engineer",
    "applied scientist",
    "applied ml",
    "search",
    "ranking",
    "retrieval",
    "data scientist",
)

NEGATIVE_TITLE_KEYWORDS = (
    "hr manager",
    "human resources",
    "marketing manager",
    "content writer",
    "graphic designer",
    "accountant",
    "sales executive",
    "customer support",
    "operations manager",
    "project manager",
    "business analyst",
    "mechanical engineer",
    "civil engineer",
    "recruiter",
)

# JD disqualifier: pure research backgrounds
RESEARCH_TITLE_KEYWORDS = (
    "research scientist",
    "research engineer",
    "ai research",
    "ml research",
    "phd researcher",
    "postdoctoral",
)

# JD disqualifier: CV/speech without NLP-IR path
CV_SPEECH_SKILLS = (
    "computer vision",
    "image classification",
    "object detection",
    "speech recognition",
    "tts",
    "gan",
    "gans",
)

IR_NLP_SKILLS = (
    "nlp",
    "retrieval",
    "ranking",
    "embedding",
    "information retrieval",
    "rag",
    "llm",
    "transformer",
    "bm25",
    "vector search",
)

LANGCHAIN_ONLY_MARKERS = (
    "langchain",
    "llamaindex",
    "langgraph",
)

# Career-description markers for production IR/ranking (not just summary template)
CAREER_IR_MARKERS = (
    "hybrid retrieval",
    "learning-to-rank",
    "learning to rank",
    "ndcg",
    "mrr",
    "bm25",
    "vector search",
    "dense vector",
    "embedding drift",
    "index refresh",
    "candidate-jd",
    "candidate jd",
    "semantic search",
    "re-ranking",
    "reranking",
    "offline/online",
    "a/b test",
    "p95",
    "retrieval latency",
    "faiss",
    "pinecone",
    "milvus",
    "elasticsearch",
    "sentence-transformer",
    "sentence transformer",
    "bge",
)

SUMMARY_IR_MARKERS = (
    "search, retrieval, and ranking",
    "hybrid retrieval",
    "learning-to-rank",
    "behavioral-signal integration",
    "offline/online evaluation",
    "embedding model selection",
    "candidate-jd matching",
    "ndcg",
)

# Non-technical role descriptions that contradict AI engineer titles (honeypot signal)
NON_TECH_ROLE_MARKERS = (
    "support agent",
    "customer support",
    "sales executive",
    "accounting",
    "hr ",
    "recruiting",
    "content writing",
    "graphic design",
    "mechanical engineering design",
    "solidworks",
    "paper products",
)

CONSULTING_FIRMS = (
    "tcs",
    "tata consultancy",
    "infosys",
    "wipro",
    "accenture",
    "cognizant",
    "capgemini",
    "hcl",
    "tech mahindra",
    "mindtree",
    "ltimindtree",
    "mphasis",
)

PRODUCT_SIGNAL_COMPANIES = (
    "amazon",
    "google",
    "meta",
    "microsoft",
    "flipkart",
    "swiggy",
    "zomato",
    "ola",
    "uber",
    "razorpay",
    "phonepe",
    "paytm",
    "redrob",
    "freshworks",
    "zoho",
    "atlassian",
    "spotify",
    "netflix",
    "airbnb",
    "myntra",
    "meesho",
    "cred",
)

CORE_SKILLS = {
    "embeddings": 1.0,
    "sentence-transformers": 1.0,
    "sentence transformers": 1.0,
    "bge": 0.9,
    "e5": 0.8,
    "openai embeddings": 0.9,
    "retrieval": 1.0,
    "vector search": 1.0,
    "hybrid search": 1.0,
    "hybrid retrieval": 1.0,
    "bm25": 0.9,
    "pinecone": 0.85,
    "weaviate": 0.85,
    "qdrant": 0.85,
    "milvus": 0.85,
    "faiss": 0.85,
    "elasticsearch": 0.8,
    "opensearch": 0.8,
    "python": 0.7,
    "ndcg": 1.0,
    "mrr": 0.9,
    "map": 0.7,
    "learning-to-rank": 1.0,
    "learning to rank": 1.0,
    "ltr": 0.8,
    "ranking": 0.85,
    "re-ranking": 0.9,
    "reranking": 0.9,
    "rag": 0.75,
    "llm": 0.6,
    "fine-tuning": 0.7,
    "fine tuning": 0.7,
    "lora": 0.65,
    "qlora": 0.65,
    "xgboost": 0.6,
    "pytorch": 0.55,
    "tensorflow": 0.5,
    "nlp": 0.55,
    "information retrieval": 0.9,
}

PREFERRED_LOCATIONS = (
    "pune",
    "noida",
    "delhi",
    "gurgaon",
    "gurugram",
    "mumbai",
    "hyderabad",
    "bangalore",
    "bengaluru",
)

IDEAL_YOE_CENTER = 6.5
IDEAL_YOE_SIGMA = 2.0
MIN_VIABLE_YOE = 4.0
MAX_PREFERRED_YOE = 12.0

# Two-stage retrieval
RECALL_POOL_SIZE = 5000
TITLE_RECALL_MIN_SCORE = 0.72

# Embedding model (downloaded during precompute only)
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
EMBEDDING_BATCH_SIZE = 256

DEFAULT_JD_PATH = Path(__file__).resolve().parent.parent / "job_description.txt"
DEFAULT_ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts"
