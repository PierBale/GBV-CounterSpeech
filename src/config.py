from pathlib import Path


SRC_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SRC_DIR.parent

# Generation models
GENERATION_MODELS = (
    "llama-3.1-8b",
    #"ministral-3-8b",
    #"qwen-3.5-9b",
)

# Dataset
DATASET = "both"  # edos | multiconan | both
TARGETS = ("WOMEN",)
EDOS_SPLIT = "test"
INCLUDE_NON_SEXIST = False
LIMIT = 150  # Maximum unique hateful messages PER dataset; 0 = all.
DATASET_SAMPLE_SEED = 42

# Generation strategies
RUN_WITHOUT_RAG = True
RAG_RETRIEVERS = (
    "qwen3-emb-4b",
    "cards",
    # "bm25",  # Optional additional baseline.
)

# Counter-speech prompt
COUNTERSPEECH_LANGUAGE = "English"
COUNTERSPEECH_MAX_SENTENCES = 2
COUNTERSPEECH_CITE_SOURCE = False

# RAG
DOCUMENTS_DIR = SRC_DIR / "1_parsed_results"
INDEX_DIR = PROJECT_DIR / "data" / "indexes"
FORCE_REINDEX = False
EMBEDDING_BATCH_SIZE = 16
CHUNK_LIBRARY_PATH = PROJECT_DIR / "data" / "card_creation" / "chunks.jsonl"
# Separate from the category-retrieval index used to create cards.
CHUNK_RAG_INDEX_PATH = INDEX_DIR / "chunk_rag_qwen3_4b.faiss"
CHUNK_RAG_EMBEDDING_BATCH_SIZE = 16

# Evidence-card RAG (the same evidence prompt as paragraph RAG).
CARD_LIBRARY_PATH = PROJECT_DIR / "data" / "card_creation" / "cards.jsonl"
CARD_INDEX_PATH = PROJECT_DIR / "data" / "card_creation" / "indexes" / "cards.faiss"
CARD_EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-4B"
CARD_EMBEDDING_BATCH_SIZE = 16

# Inference
GENERATION_BATCH_SIZE = 16
DEVICE = None
DTYPE = None  # None | float32 | float16 | bfloat16
TRUST_REMOTE_CODE = True

# Generation output
GENERATION_OUTPUT_DIR = PROJECT_DIR / "data" / "generated"

# LLM judges
JUDGE_MODELS = (
    "judgelm-7b",
)
JUDGE_INPUT_PATHS = ()  # Empty = use every generated model output.
JUDGE_OUTPUT_DIR = PROJECT_DIR / "data" / "judged"
JUDGE_LIMIT = 0
JUDGE_BATCH_SIZE = 16
JUDGE_MAX_RETRIES = 5  # Additional attempts after an invalid initial response.
JUDGE_RETRY_TEMPERATURE = 0.7
JUDGE_DEVICE = None
JUDGE_DTYPE = None
JUDGE_TRUST_REMOTE_CODE = True

# Automatic evaluation. This is a separate local workflow from judge.py.
AUTOMATIC_METRICS_INPUT_PATHS = ()  # Empty = use every generated model output.
AUTOMATIC_METRICS_OUTPUT_DIR = PROJECT_DIR / "data" / "metrics"
BERTSCORE_MODEL_TYPE = None  # None = recommended English model (roberta-large).
BERTSCORE_DEVICE = None  # None = CUDA when available, otherwise CPU.
BERTSCORE_BATCH_SIZE = 16
