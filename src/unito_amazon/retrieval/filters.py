from .schema import Document


MetadataFilter = dict[str, str | int | float | bool | None]
MetadataFilters = list[MetadataFilter]


def normalize_value(value) -> str:
    return str(value).strip().lower()


def document_matches_filter(
    document: Document,
    metadata_filter: MetadataFilter,
) -> bool:
    """
    Ritorna True se il documento soddisfa tutti i campi non-null del filtro.
    """
    for field, expected_value in metadata_filter.items():
        if expected_value is None:
            continue

        actual_value = document.metadata.get(field)

        if actual_value is None:
            return False

        if normalize_value(actual_value) != normalize_value(expected_value):
            return False

    return True


def document_matches_any_filter(
    document: Document,
    metadata_filters: MetadataFilters | None,
) -> bool:
    """
    Lista di filtri = OR.
    Singolo filtro = AND sui campi non-null.
    """
    if not metadata_filters:
        return True

    return any(
        document_matches_filter(document, metadata_filter)
        for metadata_filter in metadata_filters
    )


def get_candidate_indices(
    documents: list[Document],
    metadata_filters: MetadataFilters | None = None,
) -> list[int]:
    return [
        index
        for index, document in enumerate(documents)
        if document_matches_any_filter(document, metadata_filters)
    ]