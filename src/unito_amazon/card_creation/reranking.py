"""Qwen reranker scores P(yes | instruction, query, document)."""

import gc

from .progress import report


# Chosen with scripts/diagnose_reranker_instructions.py (no human labels):
# naming "sexist speech" in the instruction made the reranker judge passages
# about sexist hate speech in general as relevant to every label (16 distinct
# chunks in the top-4 of 11 labels; own-label rank 4.1-4.4 of 11). This
# instruction leaves the category to the query (32/25 distinct chunks; own-label
# rank 2.1/2.8) and keeps the annotation criterion (useful to respond).
RERANK_INSTRUCTION = (
    "Judge whether the passage is about the specific behaviour described in the "
    "query and contains information that would help respond to it."
)


def create_reranker(config):
    if config.reranker_model == "BAAI/bge-reranker-v2-m3":
        from unito_amazon.retrieval.reranker import SequenceClassificationReranker

        return SequenceClassificationReranker(
            config.reranker_model, max_length=config.reranker_max_tokens,
            batch_size=config.reranker_batch_size, device=config.device, truncate=False,
        )
    if config.reranker_model.startswith("Qwen/Qwen3-Reranker-"):
        return QwenReranker(
            config.reranker_model, config.reranker_batch_size,
            config.reranker_max_tokens, config.device,
        )
    raise ValueError(f"Unsupported card reranker: {config.reranker_model}")


class QwenReranker:
    def __init__(
        self, model_name: str = "Qwen/Qwen3-Reranker-4B",
        batch_size: int = 4, max_tokens: int = 32768,
        device: str | None = None, instruction: str | None = None,
    ):
        report("reranker", f"Caricamento tokenizer e modello {model_name}; device={device or 'auto'}")
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.batch_size = batch_size
        # Overridable only for diagnostics; the pipeline always uses RERANK_INSTRUCTION.
        self.instruction = instruction or RERANK_INSTRUCTION
        self.max_tokens = max_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype="auto", device_map=device or "auto",
        ).eval()
        self.max_tokens = min(max_tokens, self.model.config.max_position_embeddings)
        self.yes_id = self.tokenizer.convert_tokens_to_ids("yes")
        self.no_id = self.tokenizer.convert_tokens_to_ids("no")
        # Only the two output rows are needed. Projecting the final hidden state
        # onto them in float32 avoids bf16 logit rounding (steps of ~0.125 at
        # this magnitude), which produced many tied scores in earlier runs.
        head = self.model.get_output_embeddings()
        if getattr(head, "bias", None) is not None:
            raise ValueError("Unexpected bias in the Qwen reranker output layer.")
        self.yes_no_weight = head.weight[[self.no_id, self.yes_id]].detach().float()
        self.prefix = self.tokenizer.encode(
            '<|im_start|>system\nJudge whether the Document meets the requirements '
            'based on the Query and the Instruct provided. Note that the answer '
            'can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n',
            add_special_tokens=False,
        )
        self.suffix = self.tokenizer.encode(
            "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
            add_special_tokens=False,
        )
        report("reranker", f"Modello pronto; batch_size={batch_size}, limite input={self.max_tokens} token")

    def score(self, query: str, passages: list[str]) -> list[float]:
        scores = []
        total_batches = (len(passages) + self.batch_size - 1) // self.batch_size
        for start in range(0, len(passages), self.batch_size):
            batch_number = start // self.batch_size + 1
            end = min(start + self.batch_size, len(passages))
            report(
                "reranker", f"Batch {batch_number}/{total_batches}: passaggi "
                f"{start + 1}-{end}/{len(passages)}; tokenizzazione e scoring",
            )
            features = []
            for passage in passages[start : start + self.batch_size]:
                body = self.tokenizer.encode(
                    f"<Instruct>: {self.instruction}\n<Query>: {query}\n<Document>: {passage}",
                    add_special_tokens=False,
                )
                tokens = self.prefix + body + self.suffix
                if len(tokens) > self.max_tokens:
                    raise ValueError(
                        f"Reranker input has {len(tokens)} tokens, exceeding "
                        f"{self.max_tokens}; no paragraph has been split or truncated."
                    )
                features.append({"input_ids": tokens, "attention_mask": [1] * len(tokens)})
            inputs = self.tokenizer.pad(features, padding=True, return_tensors="pt")
            inputs = {key: value.to(self.model.device) for key, value in inputs.items()}
            with self.torch.inference_mode():
                # Left padding: position -1 is the last real token of every row.
                hidden = self.model.get_decoder()(**inputs).last_hidden_state[:, -1, :]
                weight = self.yes_no_weight.to(hidden.device)
                yes_no = hidden.float() @ weight.T
                scores.extend(yes_no.softmax(dim=-1)[:, 1].cpu().tolist())
            report("reranker", f"Batch {batch_number}/{total_batches} completato; {len(scores)}/{len(passages)} punteggi")
        return scores

    def close(self) -> None:
        report("reranker", "Rilascio modello")
        self.model = None
        self.tokenizer = None
        gc.collect()
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()
        report("reranker", "Modello rilasciato")
