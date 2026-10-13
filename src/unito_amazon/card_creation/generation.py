import gc
import re

from .config import CardCreationConfig
from .cache import StageCache
from .io import append_jsonl, fingerprint, read_jsonl, write_json, write_jsonl
from .progress import report
from .prompts import SYSTEM_PROMPT, build_card_prompt
from .schema import EvidenceCard, GenerationEnvelope, RankedPassage


class GemmaCardGenerator:
    """Text-only use of Gemma's official multimodal Transformers interface."""

    def __init__(self, config: CardCreationConfig):
        report("gemma", f"Preparazione generatore: {config.generator_model}")
        import torch
        try:
            from transformers import AutoModelForMultimodalLM, AutoProcessor
        except ImportError as error:
            raise RuntimeError(
                "Gemma 4 card generation requires an updated Transformers version "
                "with AutoModelForMultimodalLM and gemma4_unified support. "
                "See card_creation/README.md and the official Gemma model card."
            ) from error

        self.torch = torch
        self.config = config
        report("gemma", "Caricamento processor e tokenizer")
        self.processor = AutoProcessor.from_pretrained(config.generator_model)
        report("gemma", f"Caricamento pesi del modello; device={config.device or 'auto'}")
        self.model = AutoModelForMultimodalLM.from_pretrained(
            config.generator_model, dtype="auto", device_map=config.device or "auto",
        ).eval()
        report("gemma", "Modello pronto")

    def generate(self, prompt: str) -> str:
        report("gemma", "Applicazione chat template e tokenizzazione del prompt")
        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "user", "content": [{"type": "text", "text": prompt}]},
        ]
        inputs = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt", enable_thinking=False,
        )
        input_length = inputs["input_ids"].shape[-1]
        text_config = getattr(self.model.config, "text_config", self.model.config)
        context_limit = getattr(text_config, "max_position_embeddings", None)
        if context_limit and input_length + self.config.generation_max_new_tokens > context_limit:
            raise ValueError(
                f"Gemma input plus output budget exceeds {context_limit} tokens; "
                "the passage has not been split or truncated."
            )
        inputs = {key: value.to(self.model.device) for key, value in inputs.items()}
        report(
            "gemma", f"Inferenza in corso: {input_length} token di input, "
            f"massimo {self.config.generation_max_new_tokens} nuovi token",
        )
        with self.torch.inference_mode():
            outputs = self.model.generate(
                **inputs, max_new_tokens=self.config.generation_max_new_tokens,
                do_sample=False,
            )
        tokenizer = getattr(self.processor, "tokenizer", self.processor)
        response = tokenizer.decode(outputs[0, input_length:], skip_special_tokens=True).strip()
        report("gemma", f"Inferenza completata: {outputs.shape[-1] - input_length} token generati")
        return response

    def close(self) -> None:
        report("gemma", "Rilascio modello e processor")
        self.model = None
        self.processor = None
        gc.collect()
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()
        report("gemma", "Modello rilasciato")


def normalize_card(
    envelope: GenerationEnvelope, passage: RankedPassage,
    config: CardCreationConfig, generation_key: str,
) -> EvidenceCard | None:
    if envelope.card is None:
        if not envelope.rejection_reason or not envelope.rejection_reason.strip():
            raise ValueError("A rejected passage must include a rejection_reason.")
        return None
    if envelope.rejection_reason is not None:
        raise ValueError("A valid card must have rejection_reason=null.")
    content = envelope.card
    if content.source_quote not in passage.chunk.text:
        raise ValueError("source_quote is not an exact contiguous substring of the passage.")
    keywords = list(dict.fromkeys(
        re.findall(r"[A-Za-z][A-Za-z'-]{3,}", content.argument.lower())
    ))[:12]
    return EvidenceCard(
        card_id="card_" + generation_key[:24],
        primary_edos_label=passage.label, edos_definition=passage.definition,
        source=passage.chunk.source, source_chunk_id=passage.chunk.chunk_id,
        source_passage=passage.chunk.text, section_path=passage.chunk.section_path,
        source_quote=content.source_quote, reasoning=content.reasoning,
        argument=content.argument, retrieval_keywords=keywords,
        retrieval={
            "embedding_model": passage.embedding_model,
            "reranker_model": passage.reranker_model,
            "dense_rank": passage.dense_rank, "dense_score": passage.dense_score,
            "reranker_rank": passage.reranker_rank,
            "reranker_score": passage.reranker_score,
        },
        generator_model=config.generator_model, generation_key=generation_key,
        validation={
            "quote_exact_match": True,
            "semantic_faithfulness": "not_reviewed",
            "edos_alignment": "not_reviewed",
        },
    )


def generate_cards(
    config: CardCreationConfig, overwrite: bool = False, cache: StageCache | None = None,
) -> None:
    cache = cache or StageCache(config)
    ticket = cache.prepare(
        "generate", files=[config.retrieved_path], force=overwrite,
        allow_partial=True, preserve_existing=True,
    )
    if ticket.mode == "cached":
        return
    report("generate", f"Caricamento e validazione passaggi rerankati: {config.retrieved_path}")
    passages = [
        RankedPassage.model_validate(record)
        for record in read_jsonl(config.retrieved_path)
    ]
    if not passages:
        raise ValueError("No reranked passages; run the retrieve stage first.")
    report("generate", f"Passaggi da elaborare: {len(passages)}; preparazione prompt e chiavi di ripresa")
    passage_ids = [(passage.label, passage.chunk.chunk_id) for passage in passages]
    if len(set(passage_ids)) != len(passage_ids):
        raise ValueError("Duplicate label/chunk pairs in the reranked passage file.")
    jobs = []
    for passage in passages:
        prompt = build_card_prompt(passage)
        key = fingerprint({
            "model": config.generator_model, "system": SYSTEM_PROMPT,
            "prompt": prompt, "max_new_tokens": config.generation_max_new_tokens,
            "retrieval": passage.model_dump(mode="json"),
        })
        jobs.append((key, passage, prompt))
    manifest_path = config.output_dir / "generation_manifest.json"
    signature = fingerprint([key for key, _, _ in jobs])
    if not overwrite and (config.cards_path.exists() or config.attempts_path.exists()):
        report("generate", "Output esistenti: verifica manifest per la ripresa")
        manifest = cache.read_manifest(manifest_path)
        if manifest is None or manifest.get("signature") != signature:
            report("generate", "Input/modello/prompt cambiati o manifest non valido: rigenerazione automatica")
            cache.invalidate("generate")
            cache.checkpoint(ticket)
            overwrite = True

    config.output_dir.mkdir(parents=True, exist_ok=True)
    if overwrite:
        report("generate", "Overwrite attivo: sostituzione delle carte e del log dei tentativi precedenti")
    if overwrite or not config.cards_path.exists():
        write_jsonl(config.cards_path, [])
    if overwrite or not config.attempts_path.exists():
        write_jsonl(config.attempts_path, [])
    write_json(manifest_path, {"signature": signature, "model": config.generator_model})
    existing_cards = [EvidenceCard.model_validate(item) for item in read_jsonl(config.cards_path)]
    report("generate", f"Carte presenti in libreria: {len(existing_cards)}; verifica tentativi precedenti")
    completed = {card.generation_key for card in existing_cards}
    if len(completed) != len(existing_cards):
        raise ValueError("Duplicate generation keys in the card library.")
    for attempt in read_jsonl(config.attempts_path):
        if attempt.get("status") == "rejected":
            completed.add(attempt["generation_key"])
        if attempt.get("status") == "accepted" and attempt["generation_key"] not in completed:
            raise ValueError("Accepted attempt has no corresponding card; output files are inconsistent.")
    remaining = [job for job in jobs if job[0] not in completed]
    report(
        "generate", f"Ripresa: {len(jobs) - len(remaining)}/{len(jobs)} passaggi gia elaborati "
        f"(carte o scarti); {len(remaining)} da elaborare",
    )
    report("generate", f"Carte: {config.cards_path}; log tentativi: {config.attempts_path}")
    if not remaining:
        report("generate", "Nessun passaggio restante: modello non caricato")
        cache.checkpoint(ticket, complete=True)
        return

    generator = GemmaCardGenerator(config)
    failed = 0
    accepted = 0
    rejected = 0
    try:
        for position, (key, passage, prompt) in enumerate(remaining, start=1):
            report(
                "generate", f"Passaggio {position}/{len(remaining)}: {passage.label}; "
                f"chunk={passage.chunk.chunk_id}; fonte={passage.chunk.source.markdown_file}",
            )
            attempt_record = {
                "generation_key": key, "label": passage.label,
                "chunk_id": passage.chunk.chunk_id, "generator_model": config.generator_model,
            }
            for retry in range(config.generation_retries + 1):
                report("generate", f"Passaggio {position}/{len(remaining)}: tentativo {retry + 1}/{config.generation_retries + 1}")
                try:
                    raw = generator.generate(prompt)
                except Exception as error:
                    append_jsonl(config.attempts_path, {
                        **attempt_record, "status": "runtime_error", "error": str(error),
                    })
                    cache.checkpoint(ticket)
                    report("generate", f"ERRORE durante l'inferenza; dettagli salvati in {config.attempts_path}")
                    raise
                report("generate", "Risposta ricevuta: validazione JSON e corrispondenza esatta della citazione")
                try:
                    envelope = GenerationEnvelope.model_validate_json(raw)
                    card = normalize_card(envelope, passage, config, key)
                except ValueError as error:
                    append_jsonl(config.attempts_path, {
                        **attempt_record, "status": "invalid", "retry": retry,
                        "error": str(error), "raw_response": raw,
                    })
                    cache.checkpoint(ticket)
                    report("generate", f"Output non valido ({type(error).__name__}); dettagli in {config.attempts_path}")
                    if retry == config.generation_retries:
                        failed += 1
                        report("generate", f"Passaggio {position}/{len(remaining)} fallito: tentativi esauriti")
                    else:
                        report("generate", "Preparazione di un nuovo tentativo con istruzione di correzione")
                        prompt = build_card_prompt(passage) + (
                            "\nYour previous output was invalid: " + str(error)
                            + "\nReturn the required JSON with an exact quote, or reject the passage."
                        )
                    continue
                if card is not None:
                    append_jsonl(config.cards_path, card.model_dump(mode="json"))
                append_jsonl(config.attempts_path, {
                    **attempt_record,
                    "status": "accepted" if card else "rejected",
                    "retry": retry, "rejection_reason": envelope.rejection_reason,
                    "raw_response": raw,
                })
                cache.checkpoint(ticket)
                if card is not None:
                    accepted += 1
                    report("generate", f"Passaggio {position}/{len(remaining)}: carta {card.card_id} salvata")
                else:
                    rejected += 1
                    report("generate", f"Passaggio {position}/{len(remaining)} rifiutato: {envelope.rejection_reason}")
                report("generate", f"Sessione: {accepted} carte create, {rejected} scarti, {failed} falliti")
                break
    finally:
        generator.close()
    report("generate", f"Riepilogo sessione: {accepted} carte create, {rejected} scarti, {failed} falliti")
    if failed:
        raise RuntimeError(
            f"{failed} passages produced invalid cards. Inspect {config.attempts_path}; "
            "rerun generate to retry these passages while retaining completed work."
        )
    cache.checkpoint(ticket, complete=True)
    report("generate", f"Libreria carte salvata: {config.cards_path}")
