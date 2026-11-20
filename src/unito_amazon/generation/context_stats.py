"""Measure the evidence context actually included in the decoder prompt."""


def measure_context(prompt: str, evidence_count: int, tokenizer=None, tokenizer_name: str | None = None,
                    context: str | None = None) -> dict:
    """The text between <evidence> tags; for prompts without the tags, `context` (the formatted
    evidence, which must appear in the prompt)."""
    opening, closing = "<evidence>\n", "\n</evidence>"
    start, end = prompt.find(opening), prompt.rfind(closing)
    if evidence_count == 0:
        context = ""
    elif 0 <= start < end:
        context = prompt[start + len(opening):end]
    elif context is not None:
        if context not in prompt:
            raise ValueError("The evidence context is not in the prompt.")
    else:
        raise ValueError("RAG prompt is missing its evidence context delimiters.")
    return {
        "evidence_count": evidence_count,
        "characters": len(context),
        "words": len(context.split()),
        "tokens": len(tokenizer.encode(context, add_special_tokens=False)) if context and tokenizer is not None
                  else (0 if not context else None),
        "tokenizer": tokenizer_name if tokenizer is not None else None,
        "scope": "formatted_evidence_context_without_outer_tags",
    }
