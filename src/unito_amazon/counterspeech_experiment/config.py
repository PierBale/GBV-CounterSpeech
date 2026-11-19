"""Settings of the zero-shot / RAG / card-RAG counter-speech comparison (with and without EDOS label)."""

from dataclasses import asdict, dataclass, field
from itertools import combinations
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CARD_RUN_DIR = PROJECT_ROOT / "data" / "card_creation_comparison_edos_v2_words_clean_v4_20261104"

# Fixed order of the systems in the tables. *_label strategies state the EDOS label predicted
# by the classifier in the prompt and restrict retrieval to the label's pool; *_gold_label use
# the gold EDOS annotation, which exists only for EDOS.
STRATEGIES = ("zero_shot", "rag", "rag_label", "rag_gold_label", "card_rag", "card_rag_label",
              "card_rag_gold_label")
GOLD_LABEL_STRATEGIES = {"rag_gold_label", "card_rag_gold_label"}

# Pairs judged by JudgeLM, (A, B) with A shown first in the a_first order: every pair of the
# first five systems, plus RAG vs card-RAG with the same label (same pool of 15 chunks per
# label, so the pair differs only in the evidence format: raw chunk vs card).
JUDGED_PAIRS = (*combinations(("zero_shot", "rag", "card_rag", "card_rag_label", "card_rag_gold_label"), 2),
                ("rag_label", "card_rag_label"), ("rag_gold_label", "card_rag_gold_label"))


def strategies_for(source: str, strategies: tuple[str, ...] = STRATEGIES) -> tuple[str, ...]:
    """Strategies applicable to a dataset: gold-label ones need EDOS annotations."""
    return tuple(s for s in strategies if source == "edos" or s not in GOLD_LABEL_STRATEGIES)


def pairs_for(source: str, strategies: tuple[str, ...] = STRATEGIES) -> list[tuple[str, str]]:
    """Judged pairs whose two systems both apply to the dataset."""
    applicable = set(strategies_for(source, strategies))
    return [(a, b) for a, b in JUDGED_PAIRS if a in applicable and b in applicable]


@dataclass(frozen=True)
class ExperimentConfig:
    output_dir: Path = PROJECT_ROOT / "data" / "counterspeech_v2"

    # Data: unique hateful messages, sampled per dataset.
    datasets: tuple[str, ...] = ("edos", "multiconan")
    sample_size: int | None = None  # None = every unique message (EDOS test: 970, MultiCONAN: 560)
    sample_seed: int = 42
    edos_split: str = "test"

    # EDOS Task C labels predicted by the official Khan et al. M7-FE checkpoints
    # (EDOS Classifier/run_official_task_c.sh with OUTPUT_ROOT=classifier_dir).
    classifier_dir: Path = PROJECT_ROOT / "data" / "counterspeech_v2" / "edos_classifier"

    # Retrieval corpora: the v4 chunks the cards were made from, and the GPT-6.1 Sol cards.
    chunks_path: Path = CARD_RUN_DIR / "chunks.jsonl"
    cards_path: Path = (CARD_RUN_DIR / "generation"
                        / "gpt-6.1-sol__qwen3-embedding-8b__qwen3-reranker-8b__top15__prompt-v4"
                        / "cards.jsonl")
    include_section_titles: bool = True  # chunk text as seen when the cards were generated
    embedding_model: str = "Qwen/Qwen3-Embedding-8B"
    embedding_batch_size: int = 4
    embedding_max_tokens: int = 8192
    top_k: int = 5

    # Counter-speech generation.
    generator_model: str = "google/gemma-4-E4B-it"
    generator_dtype: str = "bfloat16"
    generation_batch_size: int = 16
    max_new_tokens: int = 256
    language: str = "English"
    max_sentences: int = 2

    # Pairwise JudgeLM evaluation, every pair of strategies in both presentation orders.
    judge_model: str = "BAAI/JudgeLM-7B-v1.0"
    judge_dtype: str | None = None
    judge_batch_size: int = 16
    judge_max_retries: int = 5
    judge_retry_temperature: float = 0.7

    device: str | None = None
    strategies: tuple[str, ...] = field(default=STRATEGIES)

    @property
    def subset_path(self) -> Path:
        return self.output_dir / "subset.json"

    @property
    def index_dir(self) -> Path:
        return self.output_dir / "indexes"

    @property
    def generations_path(self) -> Path:
        return self.output_dir / "generations.jsonl"

    @property
    def judgments_path(self) -> Path:
        return self.output_dir / "judgments.jsonl"

    def as_dict(self) -> dict:
        return {key: str(value) if isinstance(value, Path) else value
                for key, value in asdict(self).items()}
