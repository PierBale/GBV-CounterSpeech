.PHONY: test mock extract-mock validate-sheet mock-retrieval clean

test:
	python -m py_compile scripts/*.py src/card_routed_rag/*.py

mock:
	python scripts/03_extract_candidate_cards.py --chunks examples/example_chunks.jsonl --output data/cards/candidates/mock_candidate_cards.jsonl --mock --cards-per-label 1
	python scripts/04_normalize_cards.py --input data/cards/candidates/mock_candidate_cards.jsonl --output data/cards/candidates/mock_candidate_cards_normalized.jsonl --dedupe
	python scripts/05_prepare_expert_validation.py --input data/cards/candidates/mock_candidate_cards_normalized.jsonl --output data/validation/mock_expert_validation_sheet.csv

extract-mock:
	python scripts/03_extract_candidate_cards.py --chunks examples/example_chunks.jsonl --output data/cards/candidates/mock_candidate_cards.jsonl --mock --cards-per-label 1

validate-sheet:
	python scripts/05_prepare_expert_validation.py --input data/cards/candidates/candidate_cards_normalized.jsonl --output data/validation/expert_validation_sheet.csv

mock-retrieval:
	python scripts/08_retrieve_cards.py --cards examples/example_validated_cards.jsonl --method all --hate-speech "Women are too emotional to make rational decisions." --edos-label "3.2 immutable gender differences and gender stereotypes" --top-k 3 --output data/retrieval_outputs/mock_all_methods.json

clean:
	rm -rf data/processed/*.jsonl data/cards/candidates/*.jsonl data/cards/validated/*.jsonl data/validation/*.csv data/retrieval_outputs/*.json
