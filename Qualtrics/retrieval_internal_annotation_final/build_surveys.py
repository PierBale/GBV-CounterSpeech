"""Build the four final context-relevance questionnaires without Qualtrics randomization."""

import argparse
import copy
import csv
import hashlib
import html
import json
import random
import re
from collections import defaultdict
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RUN_ID = "edos-v2-words-clean-v4b-20261104"
INPUT_DIR = ROOT / "data/card_creation_comparison_edos_v2_words_clean_v4_20261104/runs" / RUN_ID / "reranking"
ANNOTATORS = ("marem", "arianna", "marta")
TOP_K = 4
COMMON_SIZE = 27
SEED = 20261104
MAP_FIELDS = [
    "question_id", "qualtrics_question_id", "data_export_tag", "notes_question_id",
    "notes_export_tag", "survey_id", "position", "pipeline_id", "edos_label",
    "definition", "reranker_rank", "chunk_id", "query_id", "document_source",
    "embedding_model", "reranker_model", "run_id", "query_text", "source_matches",
]


def label_key(label):
    match = re.match(r"(\d+)\.(\d+)\b", label)
    if not match:
        raise ValueError(f"Invalid EDOS label: {label}")
    return tuple(map(int, match.groups()))


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_pool(input_dir):
    paths = sorted(input_dir.glob("*.jsonl"))
    if len(paths) != 4:
        raise ValueError("Expected four configuration files")
    merged, query_details, pipeline_queries = {}, {}, []
    source_hashes = {}
    for path in paths:
        groups = defaultdict(list)
        source_path = path.resolve().relative_to(ROOT).as_posix()
        source_hashes[source_path] = file_hash(path)
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row["pipeline_id"] != path.stem or row["run_id"] != RUN_ID:
                raise ValueError(f"Unexpected configuration or run in {path}")
            qid = row["query_id"]
            details = row["label"], row["definition"], row["query_text"]
            if query_details.setdefault(qid, details) != details:
                raise ValueError(f"Conflicting query: {qid}")
            groups[qid].append(row)
        pipeline_queries.append(set(groups))
        for qid, rows in groups.items():
            ranks = [r["reranker_rank"] for r in rows]
            if any(type(rank) is not int or rank < 1 for rank in ranks):
                raise ValueError("Ranks must be positive integers")
            if len(set(ranks)) != len(ranks) or len({r["chunk_id"] for r in rows}) != len(rows):
                raise ValueError("Duplicate rank or chunk within a query/configuration")
            selected = sorted(rows, key=lambda r: r["reranker_rank"])[:TOP_K]
            if [r["reranker_rank"] for r in selected] != list(range(1, TOP_K + 1)):
                raise ValueError("Missing top-four positions")
            for row in selected:
                key = qid, row["chunk_id"]
                identity = hashlib.sha256((qid + "\0" + row["chunk_id"]).encode()).hexdigest()[:16]
                item = merged.setdefault(key, {
                    "question_id": f"Q{identity}-{row['chunk_id']}",
                    "row": row, "source_matches": [],
                })
                for field in ("text", "label", "definition", "query_text", "document_source"):
                    if item["row"][field] != row[field]:
                        raise ValueError(f"Conflicting shared item: {key}, {field}")
                item["source_matches"].append({
                    "pipeline_id": path.stem, "source_file": source_path,
                    "embedding_model": row["embedding_model"], "reranker_model": row["reranker_model"],
                    "reranker_rank": row["reranker_rank"], "reranker_score": row["reranker_score"],
                })
    if len(query_details) != 11 or any(q != set(query_details) for q in pipeline_queries):
        raise ValueError("Every configuration must cover the same 11 queries")
    if len({label_key(d[0]) for d in query_details.values()}) != 11:
        raise ValueError("Expected one query per EDOS label")
    items = sorted(merged.values(), key=lambda x: (label_key(x["row"]["label"]), x["question_id"]))
    if len(items) != 111 or sum(len(x["source_matches"]) for x in items) != 176:
        raise ValueError("Expected 111 unique query/chunk items from 176 top-four selections")
    return items, source_hashes


def split_pool(items, seed=SEED):
    """Stratify the shared sample and balance private pools within every label."""
    rng = random.Random(seed)
    groups = defaultdict(list)
    for item in items:
        groups[item["row"]["label"]].append(item)
    labels = sorted(groups, key=label_key)
    # Largest-remainder allocation: 27 shared items proportional to label pool sizes.
    quotas = {label: len(groups[label]) * COMMON_SIZE // len(items) for label in labels}
    tie_breaks = {label: rng.random() for label in labels}
    remainders = sorted(labels, key=lambda label: (
        -(len(groups[label]) * COMMON_SIZE % len(items)), tie_breaks[label]))
    for label in remainders[:COMMON_SIZE - sum(quotas.values())]:
        quotas[label] += 1
    pools = {name: [] for name in ("comune", *ANNOTATORS)}
    remaining = {}
    for label in labels:
        group = groups[label].copy()
        rng.shuffle(group)
        pools["comune"].extend(group[:quotas[label]])
        remaining[label] = group[quotas[label]:]
    capacity = {name: (len(items) - COMMON_SIZE) // 3 for name in ANNOTATORS}
    extras = {}
    for label in labels:
        group = remaining[label]
        base = len(group) // 3
        for i, name in enumerate(ANNOTATORS):
            pools[name].extend(group[i * base:(i + 1) * base])
            capacity[name] -= base
        extras[label] = group[3 * base:]
    for label in labels:
        names = list(ANNOTATORS)
        rng.shuffle(names)
        names.sort(key=lambda name: -capacity[name])
        for item, name in zip(extras[label], names):
            if capacity[name] < 1:
                raise ValueError("Cannot balance private pools")
            pools[name].append(item)
            capacity[name] -= 1
    if any(capacity.values()):
        raise ValueError("Private pool capacities were not filled")
    # Randomize offline, independently for each survey, but keep EDOS groups ordered.
    for name, pool in pools.items():
        order_rng = random.Random(f"{seed}:presentation:{name}")
        ordered = []
        for label in labels:
            group = [item for item in pool if item["row"]["label"] == label]
            group.sort(key=lambda item: item["question_id"])
            order_rng.shuffle(group)
            ordered.extend(group)
        pools[name] = ordered
    return pools


# Source markup allowed outside tables; everything else is shown as text.
ALLOWED_INLINE_TAG = re.compile(r"&lt;(/?)(sup|sub|br)\s*/?&gt;", re.I)


def escape_text(part):
    """Escape chunk text (a literal "<5%" must not open a tag), keeping sup/sub/br."""
    return ALLOWED_INLINE_TAG.sub(r"<\1\2>", html.escape(part, quote=False))


def render_context(row):
    section = re.match(r"^Section:[ \t]*([^\n]*)\n+", row["text"])
    text = row["text"][section.end():] if section else row["text"]
    heading = escape_text(section[1]) + "<br><br>" if section else ""
    parts = re.split(r"(<table\b.*?</table>)", text, flags=re.S | re.I)
    paragraph = heading + "".join(part if part.lower().startswith("<table") else
                                  '<div style="white-space: pre-wrap">' + escape_text(part) + '</div>'
                                  for part in parts if part)
    return (f'<p><strong>EDOS Label</strong></p><p><strong>{html.escape(row["label"])}</strong>'
            f' – {html.escape(row["definition"])}</p><br>'
            f'<p><strong>Paragraph</strong></p>{paragraph}<br>')


def build_survey(template, name, items):
    survey = copy.deepcopy(template)
    original = {e["PrimaryAttribute"]: e for e in survey["SurveyElements"] if e["Element"] == "SQ"}
    if not {"QID7", "QID29", "QID4"} <= original.keys():
        raise ValueError("Draft must contain task, relevance, and notes questions")
    questions, blocks, flow, mapping = [], {}, [], []
    next_id = max(int(qid[3:]) for qid in original) + 1

    def add_question(template_id, tag, text=None):
        nonlocal next_id
        question = copy.deepcopy(original[template_id])
        qid = f"QID{next_id}"
        next_id += 1
        question.update(PrimaryAttribute=qid, SecondaryAttribute=tag)
        payload = question["Payload"]
        payload.update(QuestionID=qid, DataExportTag=tag, QuestionDescription=tag)
        if text is not None:
            payload["QuestionText"] = text
        questions.append(question)
        return qid, payload

    def add_block(description, question_ids):
        index = len(blocks) + 1
        bid = f"BL_final_{index:04}"
        blocks[str(index - 1)] = {
            "Type": "Standard", "Description": description, "ID": bid,
            "BlockElements": [{"Type": "Question", "QuestionID": qid} for qid in question_ids],
            "Options": {"BlockLocking": "false", "RandomizeQuestions": "false", "BlockVisibility": "Expanded"},
        }
        flow.append({"ID": bid, "Type": "Standard", "FlowID": f"FL_{index + 1}"})

    intro, _ = add_question("QID7", "Task_description")
    intro_ids = [intro]
    if name == "comune":
        qid, payload = add_question("QID29", "Annotator", "<p><strong>Annotatrice</strong></p><p>Seleziona il tuo nome.</p>")
        payload["Choices"] = {str(i): {"Display": annotator} for i, annotator in enumerate(ANNOTATORS, 1)}
        payload["ChoiceOrder"] = ["1", "2", "3"]
        intro_ids.append(qid)
    add_block("Task description", intro_ids)
    for position, item in enumerate(items, 1):
        row, matches = item["row"], item["source_matches"]
        tag = item["question_id"]
        rating_id, _ = add_question("QID29", tag, render_context(row))
        notes_tag = "Notes_" + tag.split("-", 1)[0]
        notes_id, _ = add_question("QID4", notes_tag)
        add_block(f"{row['label']} | {tag}", [rating_id, notes_id])
        shared = len(matches) > 1
        mapping.append({
            "question_id": tag, "qualtrics_question_id": rating_id, "data_export_tag": tag,
            "notes_question_id": notes_id, "notes_export_tag": notes_tag,
            "survey_id": name, "position": position,
            "pipeline_id": ("all" if len(matches) == 4 else "shared") if shared else matches[0]["pipeline_id"],
            "edos_label": row["label"], "definition": row["definition"],
            "reranker_rank": json.dumps({m["pipeline_id"]: m["reranker_rank"] for m in matches}) if shared else matches[0]["reranker_rank"],
            "chunk_id": row["chunk_id"], "query_id": row["query_id"],
            "document_source": row["document_source"], "run_id": row["run_id"], "query_text": row["query_text"],
            "embedding_model": json.dumps(sorted({m["embedding_model"] for m in matches})) if shared else matches[0]["embedding_model"],
            "reranker_model": json.dumps(sorted({m["reranker_model"] for m in matches})) if shared else matches[0]["reranker_model"],
            "source_matches": json.dumps(matches, ensure_ascii=False),
        })
    title = f"retrieval_internal_annotation_final_{name}"
    survey["SurveyEntry"]["SurveyName"] = title
    survey["SurveyElements"] = [e for e in survey["SurveyElements"] if e["Element"] != "SQ"] + questions
    for kind in ("BL", "FL", "QC", "SO"):
        if sum(e["Element"] == kind for e in survey["SurveyElements"]) != 1:
            raise ValueError(f"Expected exactly one {kind} element")
    for element in survey["SurveyElements"]:
        if element["Element"] == "BL":
            element["Payload"] = blocks
        elif element["Element"] == "FL":
            element["Payload"] = {"Flow": flow, "Properties": {"Count": len(flow) + 1}, "FlowID": "FL_1", "Type": "Root"}
        elif element["Element"] == "QC":
            element["SecondaryAttribute"] = str(len(questions))
        elif element["Element"] == "SO":
            element["Payload"]["SurveyTitle"] = title
    return survey, mapping


def build_package(input_dir=INPUT_DIR, template_path=HERE / "Draft-retrieval-setting.qsf", seed=SEED):
    items, source_hashes = load_pool(input_dir)
    pools = split_pool(items, seed)
    template = json.loads(template_path.read_text(encoding="utf-8"))
    surveys, mappings = {}, {}
    for name, pool in pools.items():
        surveys[name], mappings[name] = build_survey(template, name, pool)
    all_ids = [r["question_id"] for rows in mappings.values() for r in rows]
    if len(all_ids) != 111 or len(set(all_ids)) != 111:
        raise ValueError("Every item must belong to exactly one of the four questionnaires")
    if [len(mappings[name]) for name in ("comune", *ANNOTATORS)] != [27, 28, 28, 28]:
        raise ValueError("Expected one common pool of 27 and three private pools of 28")
    label_counts = {}
    for label in sorted({x["row"]["label"] for x in items}, key=label_key):
        counts = {name: sum(r["edos_label"] == label for r in rows) for name, rows in mappings.items()}
        if max(counts[n] for n in ANNOTATORS) - min(counts[n] for n in ANNOTATORS) > 1:
            raise ValueError("Private pools are unbalanced within a label")
        label_counts[label] = counts
    manifest = {
        "run_id": RUN_ID, "top_k": TOP_K, "n_configurations": 4, "n_labels": 11,
        "n_raw_selections": 176, "n_unique_items": 111,
        "deduplication_key": ["query_id", "chunk_id"], "dimension": "context_relevance",
        "rating_scale": [0, 1, 2], "seed": seed, "common_size": 27,
        "common_percentage": 100 * 27 / 111, "private_size_per_annotator": 28,
        "ratings_per_annotator": 55, "total_expected_ratings": 165,
        "common_sampling": "Proportional allocation by EDOS label using largest remainders; seeded sampling within each label.",
        "private_sampling": "Disjoint pools, equal total size, per-label counts differing by at most one.",
        "presentation": "EDOS labels in numeric order, items shuffled offline within each label; no Qualtrics randomizer.",
        "input_files_sha256": source_hashes, "template_sha256": file_hash(template_path),
        "builder_sha256": file_hash(Path(__file__)), "counts_by_label": label_counts,
        "surveys": {
            name: {"qsf": f"{name}.qsf", "question_map": f"{name}_question_map.csv",
                   "n_items": len(rows), "annotators": list(ANNOTATORS) if name == "comune" else [name],
                   "items": [{**row, "source_matches": json.loads(row["source_matches"])} for row in rows]}
            for name, rows in mappings.items()
        },
    }
    return surveys, mappings, manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=INPUT_DIR)
    parser.add_argument("--template", type=Path, default=HERE / "Draft-retrieval-setting.qsf")
    parser.add_argument("--output-dir", type=Path, default=HERE)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--overwrite", action="store_true", help="Replace generated files if their contents differ")
    args = parser.parse_args()
    surveys, mappings, manifest = build_package(args.input_dir, args.template, args.seed)
    outputs = {}
    for name, survey in surveys.items():
        outputs[f"{name}.qsf"] = json.dumps(survey, ensure_ascii=False, indent=2) + "\n"
        import io
        stream = io.StringIO(newline="")
        writer = csv.DictWriter(stream, fieldnames=MAP_FIELDS)
        writer.writeheader()
        writer.writerows(mappings[name])
        outputs[f"{name}_question_map.csv"] = stream.getvalue()
    manifest["output_files_sha256"] = {name: hashlib.sha256(content.encode()).hexdigest() for name, content in outputs.items()}
    outputs["annotation_assignment.json"] = json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    for name, content in outputs.items():
        path = args.output_dir / name
        if path.exists() and path.read_bytes() != content.encode() and not args.overwrite:
            parser.error(f"{path} differs; use --overwrite only when replacing the generated questionnaires is intended")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, content in outputs.items():
        (args.output_dir / name).write_bytes(content.encode())
    for name in surveys:
        print(f"{name}.qsf: {len(mappings[name])} items, each with relevance and optional notes")
    print("111 unique items; 27 common + 28 per annotator; 55 judgments each, 165 in total.")


if __name__ == "__main__":
    main()
