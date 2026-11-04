import csv
import importlib.util
import json
from itertools import product
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch


REPO = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("rebuild", REPO / "scripts/rebuild_retrieval.py")
rebuild = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rebuild)
from unito_amazon.card_creation.config import CardCreationConfig
from unito_amazon.card_creation.queries import load_queries


class RebuildTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.output = self.root / "data/new-experiment"
        self.config = CardCreationConfig(
            output_dir=self.output, pdf_dir=self.output / "inputs/pdfs",
            markdown_dir=self.output / "markdown", labels_path=self.output / "inputs/edos_labels.json",
            run_id="test-v2", all_combinations=True, generate_evidence=False,
        )
        self.write("documents/en_doc_1.pdf", "original pdf")
        self.write("documents/en_doc_2.pdf", "new pdf")
        self.write("src/1_parsed_results/en_doc_1.md", "# Existing\n\nOld parsed text.")
        self.write("documents/metadata.xlsx", "unchanged metadata")
        old_labels = {"1.1 threats of harm": "Old first definition", "2.1 descriptive attacks": "Old second definition"}
        old_path = self.write("old-labels.json", json.dumps(old_labels))
        old_queries = [q.model_dump(mode="json") for q in load_queries(CardCreationConfig(labels_path=old_path))]
        self.write("data/card_creation_comparison/runs/confronto-02/manifest.json",
                   json.dumps({"run_id": "confronto-02", "inputs": {"queries": old_queries}}))
        self.old_cache = self.write("data/card_creation_comparison/work/index.bin", "previous cache")
        self.write("src/unito_amazon/card_creation/edos_labels.json",
                   json.dumps({label: value.replace("Old", "Corrected") for label, value in old_labels.items()}))
        for name in ("Draft-retrieval-setting.qsf", "build_retrieval_survey.py"):
            destination = self.root / "Qualtrics" / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(REPO / "Qualtrics" / name, destination)

    def write(self, name, content):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path

    def test_prepare_is_idempotent_and_preserves_old_files(self):
        manifest = rebuild.prepare(self.config, self.root)
        self.assertEqual(manifest["pdf_count"], 2)
        self.assertEqual(manifest["reused_markdown_count"], 1)
        self.assertEqual(manifest["pending_parse"], ["en_doc_2.pdf"])
        self.assertEqual(rebuild.prepare(self.config, self.root), manifest)
        self.assertEqual(self.old_cache.read_text(), "previous cache")
        self.assertEqual((self.config.markdown_dir / "en_doc_1.md").read_bytes(),
                         (self.root / "src/1_parsed_results/en_doc_1.md").read_bytes())
        self.assertFalse((self.output / "runs").exists())

    def test_refuses_overlapping_output(self):
        for path in ("data", "data/card_creation_comparison", "data/card_creation_comparison/another-run", "src/new"):
            with self.subTest(path=path), self.assertRaisesRegex(ValueError, "overlaps"):
                rebuild.check_isolation(self.config.model_copy(update={"output_dir": self.root / path}), self.root)

    def test_refuses_snapshot_replacement(self):
        source = self.write("input.bin", "first")
        destination = self.root / "snapshot.bin"
        rebuild.copy_once(source, destination)
        source.write_text("second", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Snapshot differs"):
            rebuild.copy_once(source, destination)
        self.assertEqual(destination.read_text(), "first")

    def test_detects_modified_previous_cache(self):
        rebuild.prepare(self.config, self.root)
        self.old_cache.write_text("changed", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Missing or changed"):
            rebuild.verify(self.config, self.root)

    def test_detects_tampered_frozen_pdf(self):
        rebuild.prepare(self.config, self.root)
        (self.config.pdf_dir / "en_doc_2.pdf").write_text("changed", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Missing or changed"):
            rebuild.verify(self.config, self.root)

    def test_incomplete_corpus_is_rejected_before_model_loading(self):
        rebuild.prepare(self.config, self.root)
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            rebuild.rank(self.config, self.root)

    def test_parse_uploads_only_missing_pdf_and_reuses_completed_parsing(self):
        rebuild.prepare(self.config, self.root)
        client = Mock()
        client.files.create.return_value = SimpleNamespace(id="file-2")
        client.parsing.parse.return_value = SimpleNamespace(markdown_full="New parsed text", job=SimpleNamespace(id="job-2"))
        factory = Mock(return_value=client)
        with patch.dict(rebuild.sys.modules, {
            "llama_cloud": SimpleNamespace(LlamaCloud=factory),
            "dotenv": SimpleNamespace(load_dotenv=Mock()),
        }), patch.dict(os.environ, {self.config.llama_api_key_env: "test-key"}):
            rebuild.parse(self.config, self.root)
            rebuild.parse(self.config, self.root)
        factory.assert_called_once()
        client.files.create.assert_called_once_with(file=str(self.config.pdf_dir / "en_doc_2.pdf"), purpose="parse")
        self.assertTrue((self.output / "parsed_corpus_manifest.json").is_file())
        rebuild.verify(self.config, self.root)

    def test_sealed_parsed_corpus_detects_later_changes(self):
        manifest = rebuild.prepare(self.config, self.root)
        new_md = self.write("data/new-experiment/markdown/en_doc_2.md", "New parsed text")
        rebuild.seal_markdown(self.config, manifest, self.root)
        new_md.write_text("different parsing", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Missing or changed"):
            rebuild.verify(self.config, self.root)

    def test_manifests_are_portable_between_checkouts(self):
        rebuild.prepare(self.config, self.root)
        with tempfile.TemporaryDirectory() as temporary:
            moved = Path(temporary) / "cluster-checkout"
            shutil.copytree(self.root, moved)
            portable = rebuild.portable_config(self.config, self.root)
            for key in rebuild.PATH_FIELDS:
                if portable[key] is not None:
                    portable[key] = moved / portable[key]
            rebuild.verify(CardCreationConfig(**portable), moved)

    def test_survey_rejects_incomplete_run(self):
        rebuild.prepare(self.config, self.root)
        queries = [q.model_dump(mode="json") for q in load_queries(self.config)]
        self.write("data/new-experiment/runs/test-v2/manifest.json", json.dumps({
            "status": "failed", "run_id": "test-v2", "inputs": {"queries": queries}}))
        with self.assertRaisesRegex(ValueError, "completed new run"):
            rebuild.survey(self.config, self.root)

    def test_new_survey_uses_corrected_definitions_and_all_shared_ranks(self):
        rebuild.prepare(self.config, self.root)
        queries = [q.model_dump(mode="json") for q in load_queries(self.config)]
        source_rows = []
        states = {}
        for index, (embedding, reranker) in enumerate(product(rebuild.EMBEDDING_MODELS, rebuild.RERANKING_MODELS)):
            pipeline = f"{rebuild.model_slug(embedding)}__{rebuild.model_slug(reranker)}"
            rows = []
            for query in queries:
                for rank in range(1, 4):
                    row = {"label": query["edos_label"], "definition": query["definition"],
                           "query_id": query["query_id"], "query_text": query["query_text"],
                           "chunk_id": f"chunk{(rank + index) % 3}", "reranker_rank": rank,
                           "reranker_score": 1 / rank, "dense_rank": rank, "dense_score": 1 / rank,
                           "text": "Paragraph with useful evidence", "pipeline_id": pipeline,
                           "embedding_model": embedding, "reranker_model": reranker,
                           "run_id": "test-v2", "source": {"markdown_file": "en_doc_1.md"}}
                    rows.append(row)
            source_rows.extend(rows)
            path = self.write(f"data/new-experiment/runs/test-v2/reranking/{pipeline}.jsonl",
                              "\n".join(json.dumps(row) for row in rows) + "\n")
            states[pipeline] = {"status": "complete", "sha256": rebuild.sha256(path)}
        self.write("data/new-experiment/runs/test-v2/manifest.json", json.dumps({
            "status": "complete", "run_id": "test-v2", "inputs": {"queries": queries}, "reranking": states}))
        rebuild.survey(self.config, self.root)
        destination = self.output / "survey"
        qsf = (destination / "Best_retrieval_config.qsf").read_text(encoding="utf-8")
        self.assertNotIn("Old first definition", qsf)
        for query in queries:
            self.assertIn(query["definition"], qsf)
        with (destination / "Best_retrieval_config_question_map.csv").open(encoding="utf-8", newline="") as stream:
            mapping = list(csv.DictReader(stream))
        self.assertEqual(len(mapping), 6)  # 24 selections, six unique query/chunk pairs.
        expected = {(r["pipeline_id"], r["query_id"], r["chunk_id"], r["reranker_rank"]) for r in source_rows}
        actual = {(m["pipeline_id"], row["query_id"], row["chunk_id"], m["reranker_rank"])
                  for row in mapping for m in json.loads(row["source_matches"])}
        self.assertEqual(actual, expected)
        self.assertFalse((self.root / "Qualtrics/Best_retrieval_config.qsf").exists())


if __name__ == "__main__":
    unittest.main()
