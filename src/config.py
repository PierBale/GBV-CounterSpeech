from pathlib import Path


SRC_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SRC_DIR.parent

# Generation models
GENERATION_MODELS = (
    "llama-3.2-1b",
    #"ministral-3-8b",
    #"qwen-3.5-9b",
)

# Dataset
DATASET = "multiconan"  # edos | multiconan | both
TARGETS = ("WOMEN",)
EDOS_SPLIT = "test"
INCLUDE_NON_SEXIST = False
LIMIT = 10  # 0 = all records

# Generation strategies
RUN_WITHOUT_RAG = True
RAG_RETRIEVERS = (
    "bm25",
    "qwen3-emb-0.6b",
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

# Inference
GENERATION_BATCH_SIZE = 1
DEVICE = None
DTYPE = None  # None | float32 | float16 | bfloat16
TRUST_REMOTE_CODE = True

# Generation output
GENERATION_OUTPUT_DIR = PROJECT_DIR / "data" / "generated"

# LLM judges
JUDGE_MODELS = (
    "llama-3.1-8b",
    "ministral-3-8b",
    "qwen-3.5-9b",
)
JUDGE_INPUT_PATHS = ()  # Empty = use every generated model output.
JUDGE_OUTPUT_DIR = PROJECT_DIR / "data" / "judged"
JUDGE_LIMIT = 0
JUDGE_BATCH_SIZE = 1
JUDGE_MAX_RETRIES = 1
JUDGE_DEVICE = None
JUDGE_DTYPE = None
JUDGE_TRUST_REMOTE_CODE = True
