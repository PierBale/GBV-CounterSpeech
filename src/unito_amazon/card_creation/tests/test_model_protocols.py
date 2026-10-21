"""Check backend protocols without importing inference dependencies or weights."""

import importlib
import sys
import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

from unito_amazon.card_creation.config import CardCreationConfig
from unito_amazon.card_creation.generation import GemmaCardGenerator
from unito_amazon.card_creation.reranking import create_reranker


class Values:
    shape = (1, 3)

    def to(self, device):
        return self

    def view(self, *args):
        return self

    def float(self):
        return self

    def cpu(self):
        return self

    def tolist(self):
        return [-6.0, 4.0]

    def __getitem__(self, key):
        return self


class ModelProtocolTests(unittest.TestCase):
    def test_bge_adapter_preserves_raw_logits_and_rejects_overlong_pairs(self):
        calls = []

        class Tokenizer:
            model_max_length = 8

            def __call__(self, text, text_pair=None, **kwargs):
                calls.append(kwargs)
                if "return_tensors" in kwargs:
                    return {"input_ids": Values()}
                return {"input_ids": list(range(20 if text_pair == "Too long" else 4))}

        model = Mock()
        model.return_value = SimpleNamespace(logits=Values())
        fake_transformers = SimpleNamespace(
            AutoTokenizer=SimpleNamespace(from_pretrained=Mock(return_value=Tokenizer())),
            AutoModelForSequenceClassification=SimpleNamespace(from_pretrained=Mock(return_value=model)),
        )
        fake_torch = SimpleNamespace(no_grad=nullcontext, cuda=SimpleNamespace(is_available=lambda: False))
        with patch.dict(sys.modules, {
            "transformers": fake_transformers, "torch": fake_torch,
            "sentence_transformers": SimpleNamespace(CrossEncoder=Mock()),
        }):
            # The actual shared implementation, not a replacement scoring adapter.
            importlib.import_module("unito_amazon.retrieval.reranker")
            reranker = create_reranker(CardCreationConfig(reranker_model="BAAI/bge-reranker-v2-m3"))
            self.assertEqual(reranker.score("Query", ["One", "Two"]), [-6.0, 4.0])
            self.assertTrue(all(call["truncation"] is False for call in calls))
            model.reset_mock()
            with self.assertRaisesRegex(ValueError, "does not split or truncate"):
                reranker.score("Query", ["Too long"])
            model.assert_not_called()
            reranker.close()

    def test_configured_text_llm_uses_chat_tokenizer_and_checks_context(self):
        processor = SimpleNamespace(
            apply_chat_template=Mock(return_value={"input_ids": Values()}),
            decode=Mock(return_value='{"reasoning":"Reason","evidence":"Evidence"}'),
        )
        model = Mock()
        model.eval.return_value = model
        model.config = SimpleNamespace(max_position_embeddings=20)
        model.device = "cpu"
        model.generate.return_value = Values()
        fake_transformers = SimpleNamespace(
            AutoTokenizer=SimpleNamespace(from_pretrained=Mock(return_value=processor)),
            AutoModelForCausalLM=SimpleNamespace(from_pretrained=Mock(return_value=model)),
        )
        fake_torch = SimpleNamespace(inference_mode=nullcontext, cuda=SimpleNamespace(is_available=lambda: False))
        with patch.dict(sys.modules, {"transformers": fake_transformers, "torch": fake_torch}):
            config = CardCreationConfig(generator_model="example/text-chat", generation_max_new_tokens=10)
            generator = GemmaCardGenerator(config)
            self.assertIn('"evidence"', generator.generate("Prompt"))
            messages = processor.apply_chat_template.call_args.args[0]
            self.assertEqual(messages[-1], {"role": "user", "content": "Prompt"})
            fake_transformers.AutoModelForCausalLM.from_pretrained.assert_called_once()
            model.generate.reset_mock()
            generator.config = config.model_copy(update={"generation_max_new_tokens": 30})
            with self.assertRaisesRegex(ValueError, "exceeds 20"):
                generator.generate("Prompt")
            model.generate.assert_not_called()
            generator.close()


if __name__ == "__main__":
    unittest.main()
