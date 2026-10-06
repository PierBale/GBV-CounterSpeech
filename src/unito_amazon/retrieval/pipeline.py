from typing import Any

from unito_crea.information_extraction.base import BaseInformationExtractor

from .base import BaseRetriever
from .reranker import BaseReranker
from .schema import Document, RetrievalResult


# ── Fallback ──────────────────────────────────────────────────────────────────

_FIELD_LABELS: dict[str, str] = {
    "regione": "situate in {v}",
    "circoscrizione": "nella circoscrizione {v}",
    "zona_altimetrica_3": "in zona {v}",
    "descrizione_polo_bdr": "specializzate in {v}",
    "dim_economica_bdr": "di dimensione economica {v}",
    "genere": "condotte da {v}",
    "giovane": "gestite da giovani",
    "diversificata": "diversificate",
    "biologica": "biologiche",
}

_GENERE_MAP = {"M": "uomini", "F": "donne"}
_FLAG_POSITIVE = {"S"}


def _describe_filter(f: dict[str, Any]) -> str:
    parts = []

    tipo = f.get("tipo_domanda")
    if tipo:
        parts.append(f"riguardo a '{tipo}'")

    for field, template in _FIELD_LABELS.items():
        if field in ("genere", "giovane"):
            continue  # gestiti insieme sotto

        v = f.get(field)
        if v is None:
            continue

        if field in ("diversificata", "biologica"):
            if str(v).upper() in _FLAG_POSITIVE:
                parts.append(template)
        else:
            parts.append(template.format(v=v))

    # ── genere + giovane ──────────────────────────────────────────────────────
    giovane = f.get("giovane")
    genere = f.get("genere")

    is_giovane = giovane is not None and str(giovane).upper() in _FLAG_POSITIVE
    genere_label = _GENERE_MAP.get(str(genere).upper()) if genere is not None else None

    if is_giovane and genere_label:
        parts.append(f"gestite da giovani {genere_label}")
    elif is_giovane:
        parts.append("gestite da giovani")
    elif genere_label:
        parts.append(f"condotte da {genere_label}")

    if not parts:
        return "con le caratteristiche richieste"

    return ", ".join(parts)


def build_fallback_context(
    metadata_filters: list[dict[str, Any]] | None,
) -> str:
    base = "Nella base di conoscenza non è presente alcuna opinione di aziende"

    if not metadata_filters:
        return base + " con le caratteristiche richieste dalla domanda."

    descriptions = [_describe_filter(f) for f in metadata_filters]
    unique = list(dict.fromkeys(descriptions))

    return base + " " + "; oppure ".join(unique) + "."


def _fallback_result(text: str, method_name: str) -> RetrievalResult:
    return RetrievalResult(
        document=Document(id="fallback", text=text),
        score=0.0,
        rank=1,
        retriever_name=method_name,
    )


class RetrievalPipeline:
    """
    Pipeline componibile per esperimenti di retrieval.

    Supporta:
    - retrieval puro
    - retrieval + reranking
    - metadata-based retrieval tramite metadata filter
    - metadata-based retrieval + reranking
    """

    def __init__(
        self,
        retriever: BaseRetriever,
        candidate_k: int = 30,
        final_k: int = 5,
        reranker: BaseReranker | None = None,
        metadata_extractor: BaseInformationExtractor | None = None,
        use_metadata_filter: bool = False,
        retriever_supports_metadata_filter: bool = False,
        method_name: str | None = None,
    ):
        self.retriever = retriever
        self.reranker = reranker
        self.metadata_extractor = metadata_extractor

        self.candidate_k = candidate_k
        self.final_k = final_k

        self.use_metadata_filter = use_metadata_filter
        self.retriever_supports_metadata_filter = retriever_supports_metadata_filter

        self.method_name = method_name or self._build_method_name()

    def retrieve(self, query: str) -> list[RetrievalResult]:
        _, final = self.retrieve_with_candidates(query)
        return final

    def retrieve_with_candidates(
        self, query: str
    ) -> tuple[list[RetrievalResult], list[RetrievalResult]]:
        """
        Restituisce (candidates, final_results).

        candidates: top-candidate_k risultati prima del reranking/slicing
        final_results: top-final_k risultati dopo il reranking/slicing
        """
        metadata_filters = self._extract_metadata_filters(query)

        candidates = self._retrieve_candidates(
            query=query,
            metadata_filters=metadata_filters,
        )

        if not candidates:
            fallback_text = build_fallback_context(metadata_filters)
            fallback = [_fallback_result(fallback_text, self.method_name)]
            return fallback, fallback

        if self.reranker is not None:
            final = self.reranker.rerank(
                query=query,
                results=candidates,
                top_k=self.final_k,
            )
        else:
            final = candidates[: self.final_k]

        return (
            self._with_method_name(candidates),
            self._with_method_name(final),
        )

    def _metadata_result_to_filters(
        self,
        metadata_result,
    ) -> list[dict]:
        perspective_filters = [
            perspective.model_dump(exclude_none=True)
            for perspective in metadata_result.tipologie_aziende
        ]

        perspective_filters = [
            perspective_filter
            for perspective_filter in perspective_filters
            if perspective_filter
        ]

        tipo_domanda_values = [
            value
            for value in metadata_result.tipo_domanda
            if value != "Altro"
        ]

        if not perspective_filters and tipo_domanda_values:
            return [
                {"tipo_domanda": tipo_domanda}
                for tipo_domanda in tipo_domanda_values
            ]

        if perspective_filters and not tipo_domanda_values:
            return perspective_filters

        if not perspective_filters and not tipo_domanda_values:
            return []

        filters = []

        for perspective_filter in perspective_filters:
            for tipo_domanda in tipo_domanda_values:
                combined_filter = dict(perspective_filter)
                combined_filter["tipo_domanda"] = tipo_domanda
                filters.append(combined_filter)

        return filters

    def _extract_metadata_filters(
        self,
        query: str,
    ) -> list[dict[str, Any]] | None:
        if not self.use_metadata_filter:
            return None

        if self.metadata_extractor is None:
            raise ValueError(
                "[RetrievalPipeline] use_metadata_filter=True richiede "
                "un metadata_extractor."
            )

        metadata_result = self.metadata_extractor.extract(query)

        metadata_filters = [
            metadata.model_dump(exclude_none=True)
            for metadata in metadata_result.tipologie_aziende
        ]

        metadata_filters = self._metadata_result_to_filters(metadata_result)

        if not metadata_filters:
            return None

        return metadata_filters

    def _retrieve_candidates(
        self,
        query: str,
        metadata_filters: list[dict[str, Any]] | None,
    ) -> list[RetrievalResult]:
        if metadata_filters is not None:
            if not self.retriever_supports_metadata_filter:
                raise ValueError(
                    "[RetrievalPipeline] Sono stati estratti metadata_filters, "
                    "ma il retriever configurato non supporta filtri metadato."
                )

            return self.retriever.retrieve(
                query=query,
                top_k=self.candidate_k,
                metadata_filters=metadata_filters,
            )

        return self.retriever.retrieve(
            query=query,
            top_k=self.candidate_k,
        )

    def _with_method_name(
        self,
        results: list[RetrievalResult],
    ) -> list[RetrievalResult]:
        updated_results = []

        for result in results:
            updated_results.append(
                RetrievalResult(
                    document=result.document,
                    score=result.score,
                    rank=result.rank,
                    retriever_name=self.method_name,
                )
            )

        return updated_results

    def _build_method_name(self) -> str:
        parts = []

        if self.use_metadata_filter:
            parts.append("metadata")

        parts.append(getattr(self.retriever, "retriever_name", "retriever"))

        if self.reranker is not None:
            parts.append(getattr(self.reranker, "reranker_name", "reranker"))

        return "+".join(parts)