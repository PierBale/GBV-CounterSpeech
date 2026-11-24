"""Settings of the zero-shot / RAG / card-RAG counter-speech comparison (with and without EDOS label)."""

from dataclasses import asdict, dataclass, field
from itertools import combinations
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CARD_RUN_DIR = PROJECT_ROOT / "data" / "card_creation_comparison_edos_v2_words_clean_v4_20261104"

# Fixed order of the systems in the tables. *_label strategies state the EDOS label predicted
# by the classifier in the prompt and restrict retrieval to the label's pool; *_gold_label use
# the gold EDOS annotation, which exists only for EDOS. hybrid_rag* show half cards and half
# chunks (ExperimentConfig.hybrid_cards + hybrid_chunks), retrieved independently.
STRATEGIES = ("zero_shot", "rag", "rag_label", "rag_gold_label", "card_rag", "card_rag_label",
              "card_rag_gold_label", "hybrid_rag", "hybrid_rag_label", "hybrid_rag_gold_label")
GOLD_LABEL_STRATEGIES = {"rag_gold_label", "card_rag_gold_label", "hybrid_rag_gold_label"}
# counterspeech_v4: no hybrid systems and no gold-label systems (the same 5 systems on both datasets).
V4_STRATEGIES = tuple(s for s in STRATEGIES if not s.startswith("hybrid_") and s not in GOLD_LABEL_STRATEGIES)

# Pairs judged by JudgeLM, (A, B) with A shown first in the a_first order: every pair of the
# first five systems, plus RAG vs card-RAG with the same label (same pool of 15 chunks per
# label, so the pair differs only in the evidence format: raw chunk vs card), plus the hybrid
# systems against zero-shot and against RAG and card-RAG with the same label (or none).
JUDGED_PAIRS = (*combinations(("zero_shot", "rag", "card_rag", "card_rag_label", "card_rag_gold_label"), 2),
                ("rag_label", "card_rag_label"), ("rag_gold_label", "card_rag_gold_label"),
                ("zero_shot", "hybrid_rag"), ("rag", "hybrid_rag"), ("card_rag", "hybrid_rag"),
                ("rag_label", "hybrid_rag_label"), ("card_rag_label", "hybrid_rag_label"),
                ("rag_gold_label", "hybrid_rag_gold_label"), ("card_rag_gold_label", "hybrid_rag_gold_label"))


def strategies_for(source: str, strategies: tuple[str, ...] = STRATEGIES) -> tuple[str, ...]:
    """Strategies applicable to a dataset: gold-label ones need EDOS annotations."""
    return tuple(s for s in strategies if source == "edos" or s not in GOLD_LABEL_STRATEGIES)


def judged_pairs(strategies: tuple[str, ...] = STRATEGIES, all_pairs: bool = False) -> list[tuple[str, str]]:
    """The pairs JudgeLM compares: every pair of the strategies (all_pairs, in STRATEGIES order),
    or those of JUDGED_PAIRS."""
    if all_pairs:
        return list(combinations([s for s in STRATEGIES if s in strategies], 2))
    return [(a, b) for a, b in JUDGED_PAIRS if a in strategies and b in strategies]


def pairs_for(source: str, strategies: tuple[str, ...] = STRATEGIES, all_pairs: bool = False) -> list[tuple[str, str]]:
    """Judged pairs whose two systems both apply to the dataset."""
    applicable = set(strategies_for(source, strategies))
    return [(a, b) for a, b in judged_pairs(strategies, all_pairs) if a in applicable and b in applicable]


# Earlier runs (to reproduce them, override these fields):
#   data/counterspeech_v2: sample_size None, top_k 5, hybrid_cards 3, hybrid_chunks 3,
#     prompt_style "moderator", judge_criteria "facts", strategies STRATEGIES, judge_all_pairs False.
#   data/counterspeech_v3: sample_size 400, top_k 4, hybrid 2 + 2, prompt_style "short",
#     judge_criteria "social_media", strategies STRATEGIES, judge_all_pairs False.
@dataclass(frozen=True)
class ExperimentConfig:
    output_dir: Path = PROJECT_ROOT / "data" / "counterspeech_v4"

    # Data: unique hateful messages, sampled per dataset (EDOS: sexist test messages;
    # MultiCONAN: Multitarget-CONAN, Fanton et al. 2021, target WOMEN).
    datasets: tuple[str, ...] = ("edos", "multiconan")
    # None = every unique message (EDOS test: 970, MultiCONAN: 560). With the same seed a smaller
    # sample is a prefix of a larger one (the 300 of v4 are the first 300 of the 400 of v3), and a
    # larger sample extends the saved one: only the new messages are generated and judged.
    sample_size: int | None = 300
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
    top_k: int = 4
    hybrid_cards: int = 2   # hybrid_rag*: top cards + top chunks, alternated in the prompt
    hybrid_chunks: int = 2

    # Counter-speech generation.
    generator_model: str = "google/gemma-4-E4B-it"
    generator_dtype: str = "bfloat16"
    generation_batch_size: int = 16
    max_new_tokens: int = 256
    language: str = "English"
    max_sentences: int = 2
    # "guidelines" (v4): counter-speech guidelines in the system prompt, HS / EVIDENCE / category in
    # the user prompt (generation/prompts.py, Guidelines*); "short" (v3): the CS generation prompts
    # of Damo et al. (2025, arXiv:2510.12316, Appendix B); "moderator" (v2): moderator prompt with
    # rules. All ask for JSON {"counter_speech": ...}.
    prompt_style: str = "guidelines"

    # Pairwise JudgeLM evaluation, every pair of strategies in both presentation orders.
    judge_model: str = "BAAI/JudgeLM-7B-v1.0"
    judge_dtype: str | None = None
    judge_batch_size: int = 16
    judge_max_retries: int = 5
    judge_retry_temperature: float = 0.7
    # JudgeLM criteria (evaluation/prompts.py): "social_media" or "facts" (the earlier run).
    judge_criteria: str = "social_media"
    # True: every pair of the strategies (v4); False: only config.JUDGED_PAIRS (v2, v3).
    judge_all_pairs: bool = True

    device: str | None = None
    strategies: tuple[str, ...] = field(default=V4_STRATEGIES)

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
