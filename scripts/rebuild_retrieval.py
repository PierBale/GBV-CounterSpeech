"""Prepare/resume the isolated EDOS v2 experiment; run heavy stages on the cluster."""

import argparse
import hashlib
import json
from itertools import product
from pathlib import Path
import shutil
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from unito_amazon.card_creation.config import EMBEDDING_MODELS, RERANKING_MODELS, load_config, model_slug
from unito_amazon.card_creation.queries import load_queries

DEFAULT_CONFIG = ROOT / "src/unito_amazon/card_creation/config.edos-v2.json"
LEGACY_OUTPUT = "data/card_creation_comparison"
LEGACY_MARKDOWN = "src/1_parsed_results"
LEGACY_RUN = "confronto-02"
PATH_FIELDS = (
    "pdf_dir", "markdown_dir", "output_dir", "labels_path",
    "source_metadata_path", "queries_path",
)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def files(directory):
    return sorted(p for p in directory.rglob("*") if p.is_file() and "__pycache__" not in p.parts)


def relative(path, root=ROOT):
    return path.resolve().relative_to(root.resolve()).as_posix()


def copy_once(source, destination):
    """Never silently replace an existing snapshot, including on a resumed prepare."""
    if destination.exists():
        if sha256(source) != sha256(destination):
            raise ValueError(f"Snapshot differs: {destination}; use a new experiment directory.")
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)


def write_once(path, value):
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != value:
            raise ValueError(f"Frozen manifest differs: {path}; use a new experiment directory.")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def portable_config(config, root=ROOT):
    data = config.model_dump(mode="json")
    for key in PATH_FIELDS:
        if data[key] is not None:
            data[key] = relative(Path(data[key]), root)
    return data


def check_isolation(config, root=ROOT):
    output = config.output_dir.resolve()
    relative(output, root)  # Keep all writable experiment paths inside this checkout.
    protected = [root / LEGACY_OUTPUT, root / LEGACY_MARKDOWN, root / "Qualtrics", root / "documents", root / "src"]
    for old in protected:
        old = old.resolve()
        if output == old or output in old.parents or old in output.parents:
            raise ValueError(f"Output directory overlaps existing data/code: {old}")
    for key in ("pdf_dir", "markdown_dir", "labels_path"):
        path = getattr(config, key).resolve()
        if output not in path.parents:
            raise ValueError(f"{key} must be inside the isolated output directory.")
    if len({config.pdf_dir.resolve(), config.markdown_dir.resolve(), config.labels_path.parent.resolve()}) != 3:
        raise ValueError("PDF, Markdown and label snapshot directories must be distinct.")
    if not config.run_id or config.run_id == LEGACY_RUN:
        raise ValueError("A new, explicit run_id is required.")
    if config.queries_path or config.source_metadata_path:
        raise ValueError("This rebuild expects generated label queries and no external metadata JSON.")
    if not config.all_combinations or config.generate_evidence:
        raise ValueError("This rebuild compares all four pipelines without evidence generation.")


def inventory(paths, root=ROOT):
    return {relative(path, root): sha256(path) for path in sorted(set(paths))}


def assert_inventory(expected, root=ROOT):
    changed = [name for name, digest in expected.items()
               if not (root / name).is_file() or sha256(root / name) != digest]
    if changed:
        raise ValueError("Missing or changed files:\n" + "\n".join(changed))


def verify(config, root=ROOT):
    check_isolation(config, root)
    manifest = json.loads((config.output_dir / "preparation_manifest.json").read_text(encoding="utf-8"))
    if manifest["configuration"] != portable_config(config, root):
        raise ValueError("Configuration changed after preparation; use a new experiment directory.")
    assert_inventory(manifest["previous_files_sha256"], root)
    assert_inventory(manifest["frozen_inputs_sha256"], root)
    assert_inventory(manifest["reused_markdown_sha256"], root)
    sealed = config.output_dir / "parsed_corpus_manifest.json"
    if sealed.exists():
        assert_inventory(json.loads(sealed.read_text(encoding="utf-8")), root)
    current_queries = [q.model_dump(mode="json") for q in load_queries(config)]
    saved_queries = json.loads((config.output_dir / "inputs/queries.json").read_text(encoding="utf-8"))
    if current_queries != saved_queries:
        raise ValueError("Query definitions differ from the prepared experiment.")
    print(f"Verified: {len(manifest['previous_files_sha256'])} previous files unchanged; frozen inputs intact.")
    return manifest


def prepare(config, root=ROOT):
    check_isolation(config, root)
    manifest_path = config.output_dir / "preparation_manifest.json"
    if manifest_path.exists():
        return verify(config, root)
    inputs = config.output_dir / "inputs"
    canonical = root / "src/unito_amazon/card_creation/edos_labels.json"
    previous_run = root / LEGACY_OUTPUT / "runs" / LEGACY_RUN / "manifest.json"
    old_manifest = json.loads(previous_run.read_text(encoding="utf-8"))
    old_queries = old_manifest["inputs"]["queries"]
    old_labels = {q["edos_label"]: q["definition"] for q in old_queries}
    previous_labels = inputs / "edos_labels.previous.json"
    if previous_labels.exists():
        if json.loads(previous_labels.read_text(encoding="utf-8")) != old_labels:
            raise ValueError("Previous label snapshot differs from the historical run.")
    else:
        write_once(previous_labels, old_labels)
    copy_once(canonical, config.labels_path)
    copy_once(previous_run, inputs / "previous_run_manifest.json")
    copy_once(root / "Qualtrics/Draft-retrieval-setting.qsf", inputs / "survey_template.qsf")
    metadata = root / "documents/metadata.xlsx"
    if metadata.exists():
        copy_once(metadata, inputs / "metadata.xlsx")
    pdfs = [p for p in files(root / "documents") if p.suffix.lower() == ".pdf"]
    if not pdfs:
        raise ValueError("No source PDFs found.")
    reused, pending = [], []
    for source in pdfs:
        name = source.relative_to(root / "documents")
        copy_once(source, config.pdf_dir / name)
        old_md = root / LEGACY_MARKDOWN / name.with_suffix(".md")
        new_md = config.markdown_dir / name.with_suffix(".md")
        if old_md.is_file() and old_md.stat().st_size:
            copy_once(old_md, new_md)
            reused.append(new_md)
            old_sidecar = old_md.with_suffix(".parse.json")
            if old_sidecar.exists():
                new_sidecar = new_md.with_suffix(".parse.json")
                copy_once(old_sidecar, new_sidecar)
                reused.append(new_sidecar)
        else:
            pending.append(name.as_posix())
    queries = [q.model_dump(mode="json") for q in load_queries(config)]
    if set(old_labels) != {q["edos_label"] for q in queries}:
        raise ValueError("EDOS category keys changed unexpectedly.")
    if {q["query_id"] for q in old_queries} & {q["query_id"] for q in queries}:
        raise ValueError("Every corrected definition must create a new query ID.")
    write_once(inputs / "queries.json", queries)
    write_once(inputs / "configuration.json", portable_config(config, root))
    old_by_label = {q["edos_label"]: q for q in old_queries}
    write_once(inputs / "query_id_migration.json", [
        {"edos_label": q["edos_label"], "old_query_id": old_by_label[q["edos_label"]]["query_id"],
         "new_query_id": q["query_id"], "definition_changed": old_labels[q["edos_label"]] != q["definition"]}
        for q in queries
    ])
    previous = files(root / LEGACY_OUTPUT) + files(root / LEGACY_MARKDOWN) + files(root / "Qualtrics")
    previous += files(root / "documents")
    manifest = {
        "prepared_at": datetime.now(timezone.utc).isoformat(), "run_id": config.run_id,
        "configuration": portable_config(config, root), "pdf_count": len(pdfs),
        "reused_markdown_count": sum(p.suffix == ".md" for p in reused),
        "pending_parse": pending,
        "previous_files_sha256": inventory(previous, root),
        "frozen_inputs_sha256": inventory(files(inputs), root),
        "reused_markdown_sha256": inventory(reused, root),
        "legacy_markdown_provenance": "Copied byte-for-byte; legacy files without parse sidecars retain unverified provenance.",
    }
    write_once(manifest_path, manifest)
    verify(config, root)
    print(f"Prepared {len(pdfs)} PDFs; reused {manifest['reused_markdown_count']} Markdown; to parse: {', '.join(pending)}")
    return manifest


def seal_markdown(config, manifest, root=ROOT):
    expected = {p.relative_to(config.pdf_dir).with_suffix(".md")
                for p in files(config.pdf_dir) if p.suffix.lower() == ".pdf"}
    actual = {p.relative_to(config.markdown_dir) for p in config.markdown_dir.rglob("*.md")}
    missing = sorted(str(p) for p in expected if p not in actual or not (config.markdown_dir / p).stat().st_size)
    if missing or actual - expected or len(expected) != manifest["pdf_count"]:
        raise ValueError(f"Incomplete/unexpected Markdown corpus; run parse first. Missing: {missing}; extra: {actual - expected}")
    write_once(config.output_dir / "parsed_corpus_manifest.json", inventory(files(config.markdown_dir), root))


def parse(config, root=ROOT):
    manifest = verify(config, root)
    from unito_amazon.card_creation.parsing import parse_pdfs
    parse_pdfs(config, overwrite=False)
    seal_markdown(config, manifest, root)
    verify(config, root)


def rank(config, root=ROOT):
    manifest = verify(config, root)
    seal_markdown(config, manifest, root)
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("Submit the rank stage to a cluster GPU node with CUDA available.")
    from unito_amazon.card_creation.experiment import run_experiment
    run_experiment(config, force_reindex=False, overwrite=False)
    verify(config, root)


def survey(config, root=ROOT):
    verify(config, root)
    run = config.output_dir / "runs" / config.run_id
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    queries = json.loads((config.output_dir / "inputs/queries.json").read_text(encoding="utf-8"))
    if manifest["status"] != "complete" or manifest["run_id"] != config.run_id or manifest["inputs"]["queries"] != queries:
        raise ValueError("Survey requires a completed new run with the corrected query definitions.")
    expected = {f"{model_slug(e)}__{model_slug(r)}" for e, r in product(EMBEDDING_MODELS, RERANKING_MODELS)}
    paths = {p.stem: p for p in (run / "reranking").glob("*.jsonl")}
    if set(paths) != expected or set(manifest["reranking"]) != expected:
        raise ValueError("Survey requires the four expected pipeline archives.")
    by_id = {q["query_id"]: q for q in queries}
    for pipeline, path in paths.items():
        state = manifest["reranking"][pipeline]
        if state["status"] != "complete" or sha256(path) != state["sha256"]:
            raise ValueError(f"Incomplete or changed ranking archive: {path}")
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            query = by_id.get(row["query_id"])
            if not query or row["run_id"] != config.run_id or any(
                row[field] != query[key]
                for field, key in (("label", "edos_label"), ("definition", "definition"), ("query_text", "query_text"))
            ):
                raise ValueError(f"Ranking row differs from the corrected queries: {path}")
    destination = config.output_dir / "survey"
    subprocess.run([
        sys.executable, str(root / "Qualtrics/build_retrieval_survey.py"),
        "--input-dir", str(run / "reranking"), "--template", str(config.output_dir / "inputs/survey_template.qsf"),
        "--top-n", "3", "--merge-common-chunks",
        "--output", str(destination / "Best_retrieval_config.qsf"),
        "--csv-output", str(destination / "Best_retrieval_config_question_map.csv"),
    ], check=True)
    verify(config, root)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "parse", "rank", "survey", "verify", "all"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config = load_config(args.config)
    stages = ("prepare", "parse", "rank", "survey", "verify") if args.stage == "all" else (args.stage,)
    for stage in stages:
        globals()[stage](config)


if __name__ == "__main__":
    main()
