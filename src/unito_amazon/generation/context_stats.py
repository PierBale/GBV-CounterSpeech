"""Measure the evidence context actually included in the decoder prompt."""


def measure_context(prompt: str, evidence_count: int, tokenizer=None, tokenizer_name: str | None = None) -> dict:
    if evidence_count == 0:
        context = ""
    else:
        opening, closing = "<evidence>\n", "\n</evidence>"
        start = prompt.find(opening)
        end = prompt.rfind(closing)
        if start < 0 or end < start:
            raise ValueError("RAG prompt is missing its evidence context delimiters.")
        context = prompt[start + len(opening):end]
    return {
        "evidence_count": evidence_count,
        "characters": len(context),
        "words": len(context.split()),
        "tokens": len(tokenizer.encode(context, add_special_tokens=False)) if context and tokenizer is not None
                  else (0 if not context else None),
        "tokenizer": tokenizer_name if tokenizer is not None else None,
        "scope": "formatted_evidence_context_without_outer_tags",
    }
