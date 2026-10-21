"""Deterministic query selection, shared by every retrieval configuration."""

import json

from .config import CardCreationConfig
from .io import fingerprint, read_jsonl
from .schema import RetrievalQuery


def load_queries(config: CardCreationConfig) -> list[RetrievalQuery]:
    labels = json.loads(config.labels_path.read_text(encoding="utf-8"))
    if not isinstance(labels, dict) or not labels or any(
        not isinstance(label, str) or not isinstance(definition, str)
        or not label.strip() or not definition.strip() for label, definition in labels.items()
    ):
        raise ValueError("Labels JSON must map each category to a non-empty definition.")
    records = read_jsonl(config.queries_path) if config.queries_path else [
        {"edos_label": label, "query_text": f"Sexism type: {label}\nDescription: {definition}"}
        for label, definition in labels.items()
    ]
    queries = []
    for record in records:
        label = record.get("edos_label")
        text = record.get("query_text")
        if label not in labels:
            raise ValueError(f"Query refers to an unknown EDOS label: {label!r}")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("query_text must be a non-empty string.")
        query = RetrievalQuery.model_validate({
            "definition": labels[label],
            "query_id": "query_" + fingerprint([label, text])[:24],
            **record,
        })
        if not query.query_id.strip() or query.definition != labels[label]:
            raise ValueError("Query ID is blank or definition differs from the EDOS labels file.")
        queries.append(query)
    if not queries or len({query.query_id for query in queries}) != len(queries):
        raise ValueError("Queries must be non-empty and have distinct query_id values.")
    return queries
