"""Automatic evaluation metrics for generated counter-speech.

BLEU-4, ROUGE-L, METEOR and BERTScore are computed through Hugging Face
``evaluate`` and are enabled only for MultiCONAN records. Distinct-n and the
Cettolo et al. repetition rate do not require a reference and are implemented
locally because Hugging Face Evaluate does not provide those metrics.
"""

from __future__ import annotations

import gc
import math
import re
from collections import Counter
from statistics import fmean
from typing import Iterable, Sequence


REFERENCE_METRIC_FIELDS = (
    "bleu_4",
    "rouge_l",
    "meteor",
    "bertscore_precision",
    "bertscore_recall",
    "bertscore_f1",
    "bertscore_rescaled_precision",
    "bertscore_rescaled_recall",
    "bertscore_rescaled_f1",
)
REFERENCE_FREE_METRIC_FIELDS = (
    "distinct_1",
    "distinct_2",
    "repetition_rate",
)
AUTOMATIC_METRIC_FIELDS = (
    *REFERENCE_METRIC_FIELDS,
    *REFERENCE_FREE_METRIC_FIELDS,
)

_WORD_PATTERN = re.compile(r"\b\w+(?:['’]\w+)?\b", flags=re.UNICODE)
_LIGHTWEIGHT_METRICS: dict[str, object] = {}


def compute_automatic_metrics(
    records: Sequence[dict],
    *,
    bertscore_model_type: str | None = None,
    bertscore_device: str | None = None,
    bertscore_batch_size: int = 16,
) -> list[dict]:
    """Attach per-generation automatic metrics to copies of ``records``."""
    if bertscore_batch_size < 1:
        raise ValueError("bertscore_batch_size must be at least 1.")

    enriched = [dict(record) for record in records]
    reference_indices: list[int] = []
    candidates: list[str] = []
    references: list[str] = []

    for index, record in enumerate(enriched):
        candidate = _normalize_text(record.get("generated_counter_speech"))
        if not candidate:
            raise ValueError(
                f"Record {record.get('id', index)!r} has no generated counter-speech."
            )

        tokens = tokenize(candidate)
        metrics = {field: None for field in REFERENCE_METRIC_FIELDS}
        metrics.update(
            {
                "distinct_1": distinct_n(tokens, 1),
                "distinct_2": distinct_n(tokens, 2),
                "repetition_rate": repetition_rate(tokens),
            }
        )

        reference = _normalize_text(record.get("reference_counter_speech"))
        if _is_multiconan(record) and reference:
            reference_indices.append(index)
            candidates.append(candidate)
            references.append(reference)

        enriched[index]["automatic_metrics"] = metrics

    if reference_indices:
        reference_scores = _reference_metrics(
            candidates,
            references,
            bertscore_model_type=bertscore_model_type,
            bertscore_device=bertscore_device,
            bertscore_batch_size=bertscore_batch_size,
        )
        for index, scores in zip(reference_indices, reference_scores, strict=True):
            enriched[index]["automatic_metrics"].update(scores)

    return enriched


def _reference_metrics(
    candidates: list[str],
    references: list[str],
    *,
    bertscore_model_type: str | None,
    bertscore_device: str | None,
    bertscore_batch_size: int,
) -> list[dict]:
    bleu = _load_metric("bleu")
    rouge = _load_metric("rouge")
    meteor = _load_metric("meteor")

    sentence_bleu = [
        bleu.compute(
            predictions=[candidate],
            references=[reference],
            max_order=4,
            smooth=True,
        )["bleu"]
        for candidate, reference in zip(candidates, references, strict=True)
    ]
    rouge_l = rouge.compute(
        predictions=candidates,
        references=references,
        rouge_types=["rougeL"],
        use_aggregator=False,
        use_stemmer=True,
    )["rougeL"]
    meteor_scores = [
        meteor.compute(predictions=[candidate], references=[reference])["meteor"]
        for candidate, reference in zip(candidates, references, strict=True)
    ]

    bertscore = _load_metric("bertscore", cache=False)
    bertscore_kwargs = {
        "predictions": candidates,
        "references": references,
        "lang": "en",
        "device": bertscore_device,
        "batch_size": bertscore_batch_size,
    }
    if bertscore_model_type:
        bertscore_kwargs["model_type"] = bertscore_model_type

    raw = bertscore.compute(
        **bertscore_kwargs,
        rescale_with_baseline=False,
    )
    rescaled = bertscore.compute(
        **bertscore_kwargs,
        rescale_with_baseline=True,
    )

    output = []
    for index in range(len(candidates)):
        output.append(
            {
                "bleu_4": _round(sentence_bleu[index]),
                "rouge_l": _round(rouge_l[index]),
                "meteor": _round(meteor_scores[index]),
                "bertscore_precision": _round(raw["precision"][index]),
                "bertscore_recall": _round(raw["recall"][index]),
                "bertscore_f1": _round(raw["f1"][index]),
                "bertscore_rescaled_precision": _round(
                    rescaled["precision"][index]
                ),
                "bertscore_rescaled_recall": _round(rescaled["recall"][index]),
                "bertscore_rescaled_f1": _round(rescaled["f1"][index]),
            }
        )

    del bertscore, raw, rescaled
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
    return output


def _load_metric(name: str, *, cache: bool = True):
    if cache and name in _LIGHTWEIGHT_METRICS:
        return _LIGHTWEIGHT_METRICS[name]
    try:
        import evaluate
    except ImportError as error:
        raise RuntimeError(
            "Automatic metrics require Hugging Face evaluate. Install "
            "requirements-generation.txt in the active environment."
        ) from error

    try:
        metric = evaluate.load(name)
    except Exception as error:
        raise RuntimeError(
            f"Unable to load the Hugging Face Evaluate metric {name!r}. "
            "The first run requires network access; later runs use the local cache."
        ) from error
    if cache:
        _LIGHTWEIGHT_METRICS[name] = metric
    return metric


def _normalize_text(value: object) -> str:
    if isinstance(value, (list, tuple)):
        value = value[0] if value else ""
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _is_multiconan(record: dict) -> bool:
    return str(record.get("source") or "").strip().casefold() == "multiconan"


def tokenize(text: str) -> list[str]:
    """Lower-case word tokenizer shared by the diversity metrics."""
    return [match.group(0).casefold() for match in _WORD_PATTERN.finditer(text)]


def ngrams(tokens: Sequence[str], n: int) -> list[tuple[str, ...]]:
    if n < 1:
        raise ValueError("n must be at least 1.")
    return [tuple(tokens[index : index + n]) for index in range(len(tokens) - n + 1)]


def distinct_n(tokens: Sequence[str], n: int) -> float | None:
    items = ngrams(tokens, n)
    if not items:
        return None
    return _round(len(set(items)) / len(items))


def repetition_rate(tokens: Sequence[str], max_n: int = 4) -> float | None:
    """Geometric mean of non-singleton n-gram type rates (Cettolo et al.)."""
    rates = []
    for n in range(1, max_n + 1):
        counts = Counter(ngrams(tokens, n))
        if not counts:
            continue
        repeated_types = sum(count > 1 for count in counts.values())
        rates.append(repeated_types / len(counts))

    if not rates:
        return None
    if any(rate == 0 for rate in rates):
        return 0.0
    return _round(math.prod(rates) ** (1 / len(rates)))


def summarize_automatic_metrics(
    records: Sequence[dict],
    *,
    compute_corpus_bleu: bool = True,
) -> dict:
    """Compute corpus-level and mean automatic metrics for a record group."""
    candidates = [
        _normalize_text(record.get("generated_counter_speech"))
        for record in records
    ]
    reference_records = [
        record
        for record in records
        if _is_multiconan(record)
        and _normalize_text(record.get("reference_counter_speech"))
        and record.get("automatic_metrics")
    ]

    return {
        "reference_based": {
            "eligible_records": len(reference_records),
            "bleu_4_corpus": (
                _corpus_bleu(reference_records) if compute_corpus_bleu else None
            ),
            "rouge_l_mean": _metric_mean(reference_records, "rouge_l"),
            "meteor_mean": _metric_mean(reference_records, "meteor"),
            "bertscore_precision_mean": _metric_mean(
                reference_records, "bertscore_precision"
            ),
            "bertscore_recall_mean": _metric_mean(
                reference_records, "bertscore_recall"
            ),
            "bertscore_f1_mean": _metric_mean(reference_records, "bertscore_f1"),
            "bertscore_rescaled_precision_mean": _metric_mean(
                reference_records, "bertscore_rescaled_precision"
            ),
            "bertscore_rescaled_recall_mean": _metric_mean(
                reference_records, "bertscore_rescaled_recall"
            ),
            "bertscore_rescaled_f1_mean": _metric_mean(
                reference_records, "bertscore_rescaled_f1"
            ),
        },
        "reference_free": {
            "records": len(records),
            "distinct_1_corpus": corpus_distinct_n(candidates, 1),
            "distinct_2_corpus": corpus_distinct_n(candidates, 2),
            "repetition_rate_mean": _metric_mean(records, "repetition_rate"),
        },
    }


def _corpus_bleu(records: Sequence[dict]) -> float | None:
    if not records:
        return None
    bleu = _load_metric("bleu")
    candidates = [
        _normalize_text(record.get("generated_counter_speech"))
        for record in records
    ]
    references = [
        _normalize_text(record.get("reference_counter_speech"))
        for record in records
    ]
    return _round(
        bleu.compute(
            predictions=candidates,
            references=references,
            max_order=4,
            smooth=False,
        )["bleu"]
    )


def corpus_distinct_n(texts: Iterable[str], n: int) -> float | None:
    all_ngrams = []
    for text in texts:
        all_ngrams.extend(ngrams(tokenize(text), n))
    if not all_ngrams:
        return None
    return _round(len(set(all_ngrams)) / len(all_ngrams))


def _metric_mean(records: Sequence[dict], field: str) -> float | None:
    values = [
        record.get("automatic_metrics", {}).get(field)
        for record in records
        if isinstance(record.get("automatic_metrics"), dict)
    ]
    numeric = [float(value) for value in values if value is not None]
    return _round(fmean(numeric)) if numeric else None


def _round(value: float, digits: int = 6) -> float:
    return round(float(value), digits)
