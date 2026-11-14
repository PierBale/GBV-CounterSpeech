"""Build four model-annotation QSFs by cloning Draft_best-model.qsf.

Run with Python's standard library only. All sampling and offline ordering are
controlled by --seed; Qualtrics receives a fixed block/question order.
"""

import argparse
import copy
import csv
import hashlib
import html
import io
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
INPUT_DIR = ROOT / "data/card_creation_comparison_edos_v2_words_clean_v4_20261104/generation"
INPUT_FILES = tuple(
    INPUT_DIR / f"{model}__qwen3-embedding-8b__qwen3-reranker-8b__top15__prompt-v4/cards.jsonl"
    for model in ("deepseek-v4.1-flash", "gpt-6.1-sol")
)
TEMPLATE = HERE / "Draft_best-model.qsf"
ANNOTATORS = ("marem", "arianna", "marta")
TOP_K = 6
N_SELECTIONS = 6 * 2 * 11  # top-6 x 2 models x 11 EDOS labels
N_ITEMS = 132  # model-specific cards; do not merge cards sharing a chunk
COMMON_SIZE = 45
PRIVATE_SIZE = (N_ITEMS - COMMON_SIZE) // 3  # 29
SEED = 20261114
MAP_FIELDS = [
    "question_id", "survey_id", "position", "card_id", "generator_model",
    "primary_edos_label", "edos_definition", "reranker_rank", "chunk_id",
    "query_id", "source_file", "source_id", "markdown_file",
    "instruction_question_id", "faithfulness_question_id", "faithfulness_export_tag",
    "answer_relevance_question_id", "answer_relevance_export_tag",
    "notes_question_id", "notes_export_tag",
]


def label_key(label):
    match = re.match(r"^(\d+)\.(\d+)\b", label)
    if not match:
        raise ValueError(f"Invalid EDOS label: {label}")
    return tuple(map(int, match.groups()))


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_name(path):
    path = path.resolve()
    return path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else str(path)


def item_id(row):
    """Export-safe logical question ID from both card_id and generator_model.

    Internal Qualtrics QuestionIDs stay numeric QID<n>; the full identity and
    the dimension appear in DataExportTag and in each question-map row.
    """
    model = re.sub(r"[^A-Za-z0-9_-]", "_", row["generator_model"])
    return f"{row['card_id']}__{model}"


def load_pool(input_files=INPUT_FILES):
    if len(input_files) != 2:
        raise ValueError("Expected exactly two model source files")
    items, hashes, models, label_sets = [], {}, set(), []
    definitions = {}
    for path in sorted(map(Path, input_files)):
        groups = defaultdict(list)
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        file_models = {row["generator_model"] for row in rows}
        if len(file_models) != 1 or models & file_models:
            raise ValueError(f"Expected one distinct generator_model per file: {path}")
        models.update(file_models)
        source = source_name(path)
        hashes[source] = file_hash(path)
        for row in rows:
            label = row["primary_edos_label"]
            label_key(label)
            if definitions.setdefault(label, row["edos_definition"]) != row["edos_definition"]:
                raise ValueError(f"Conflicting EDOS definitions: {label}")
            groups[label].append(row)
        if len(groups) != 11 or len({label_key(label) for label in groups}) != 11:
            raise ValueError(f"Expected all 11 distinct EDOS labels: {path}")
        label_sets.append(set(groups))
        for label in sorted(groups, key=label_key):
            group = groups[label]
            ranks = [row["retrieval"]["reranker_rank"] for row in group]
            if any(type(rank) is not int or rank < 1 for rank in ranks):
                raise ValueError(f"Ranks must be positive integers: {path}, {label}")
            if len(set(ranks)) != len(ranks):
                raise ValueError(f"Duplicate reranker rank: {path}, {label}")
            selected = sorted(group, key=lambda row: row["retrieval"]["reranker_rank"])[:TOP_K]
            if [row["retrieval"]["reranker_rank"] for row in selected] != list(range(1, TOP_K + 1)):
                raise ValueError(f"Missing top-{TOP_K} ranks: {path}, {label}")
            if len({row["chunk"]["chunk_id"] for row in selected}) != TOP_K:
                raise ValueError(f"Duplicate selected chunk within a model/label: {path}, {label}")
            for row in selected:
                for text in (row["evidence"], row["chunk"]["text"], row["card_id"]):
                    if not isinstance(text, str) or not text.strip():
                        raise ValueError(f"Empty card content: {path}, {label}")
                items.append({"question_id": item_id(row), "row": row, "source_file": source})
    if label_sets[0] != label_sets[1]:
        raise ValueError("Models must cover the same EDOS labels")
    if len(items) != N_SELECTIONS or len(items) != N_ITEMS:
        raise ValueError(f"Expected {N_ITEMS} selected model-specific cards, found {len(items)}")
    if len({item["question_id"] for item in items}) != N_ITEMS:
        raise ValueError("Duplicate model/card question IDs")
    return sorted(items, key=lambda item: (label_key(item["row"]["primary_edos_label"]), item["question_id"])), hashes


def split_pool(items, seed=SEED):
    """Stratify common cards by label and model; balance private label quotas."""
    if len(items) != N_ITEMS or (N_ITEMS - COMMON_SIZE) != PRIVATE_SIZE * len(ANNOTATORS):
        raise ValueError("Item count must permit exactly equal private pools")
    rng = random.Random(seed)
    groups = defaultdict(list)
    for item in items:
        groups[item["row"]["primary_edos_label"]].append(item)
    labels = sorted(groups, key=label_key)
    quotas = {label: len(groups[label]) * COMMON_SIZE // len(items) for label in labels}
    tie_breaks = {label: rng.random() for label in labels}
    remainder_order = sorted(labels, key=lambda label: (
        -(len(groups[label]) * COMMON_SIZE % len(items)), tie_breaks[label], label_key(label)))
    for label in remainder_order[:COMMON_SIZE - sum(quotas.values())]:
        quotas[label] += 1
    pools = {name: [] for name in ("comune", *ANNOTATORS)}
    remaining = {}
    for label in labels:
        by_model = defaultdict(list)
        for item in groups[label]:
            by_model[item["row"]["generator_model"]].append(item)
        models = sorted(by_model)
        if len(models) != 2 or quotas[label] not in (4, 5):
            raise ValueError("Common label quotas require two models and four or five cards")
        model_quotas = {model: 2 for model in models}
        if quotas[label] == 5:
            # The 45th card's model is an equal-probability, seeded choice.
            model_quotas[rng.choice(models)] += 1
        remaining[label] = []
        for model in models:
            group = sorted(by_model[model], key=lambda item: item["question_id"])
            size = model_quotas[model]
            if len(group) < size:
                raise ValueError(f"Not enough common cards for {label}, {model}")
            rng.shuffle(group)
            pools["comune"].extend(group[:size])
            remaining[label].extend(group[size:])
    # First reserve each label's equal base share. Assign its remaining one or
    # two cards to distinct annotators with the most capacity; shuffle ties.
    capacities = {name: PRIVATE_SIZE for name in ANNOTATORS}
    private_quotas = {}
    for label in labels:
        base = len(remaining[label]) // len(ANNOTATORS)
        private_quotas[label] = {name: base for name in ANNOTATORS}
        for name in ANNOTATORS:
            capacities[name] -= base
    for label in labels:
        names = list(ANNOTATORS)
        rng.shuffle(names)
        names.sort(key=lambda name: -capacities[name])
        for name in names[:len(remaining[label]) % len(ANNOTATORS)]:
            if capacities[name] <= 0:
                raise ValueError("Cannot balance private quotas")
            private_quotas[label][name] += 1
            capacities[name] -= 1
    if any(capacities.values()):
        raise ValueError("Private quotas do not fill the required capacities")
    for label in labels:
        group = remaining[label]
        rng.shuffle(group)
        offset = 0
        for name in ANNOTATORS:
            size = private_quotas[label][name]
            pools[name].extend(group[offset:offset + size])
            offset += size
    for name, pool in pools.items():
        order_rng = random.Random(f"{seed}:presentation:{name}")
        ordered = []
        for label in labels:
            group = sorted((item for item in pool if item["row"]["primary_edos_label"] == label),
                           key=lambda item: item["question_id"])
            order_rng.shuffle(group)
            ordered.extend(group)
        pools[name] = ordered
    return pools


def paragraph_text(row):
    # Preserve the section title/path and all body text; remove only the marker.
    return re.sub(r"\ASection:[ \t]*", "", row["chunk"]["text"], count=1)


def render_paragraph(row):
    # Keep source tables readable, and escape prose (including literal '<5%').
    # The local corpus contains HTML tables and sup/sub/br inline markup.
    parts = re.split(r"(<table\b.*?</table>)", paragraph_text(row), flags=re.S | re.I)
    rendered = []
    for part in parts:
        if not part:
            continue
        if part.lower().startswith("<table"):
            rendered.append(part)
        else:
            escaped = html.escape(part, quote=False)
            escaped = re.sub(r"&lt;(/?)(sup|sub|br)\s*/?&gt;", r"<\1\2>", escaped, flags=re.I)
            rendered.append(f'<div style="white-space: pre-wrap">{escaped}</div>')
    return "".join(rendered)


def render_rating(row, original_text, dimension):
    # Retain the draft's exact rating prompt while replacing its example texts.
    prompt = re.search(r"<strong>How relevant\b.*\Z", original_text, flags=re.S)
    if not prompt:
        raise ValueError(f"Cannot find the draft rating prompt: {dimension}")
    if dimension == "faithfulness":
        context = f'<strong>Paragraph</strong><br>{render_paragraph(row)}'
    else:
        context = (f'<hr><br><strong>EDOS Label</strong><br><p>'
                   f'{html.escape(row["primary_edos_label"])} – {html.escape(row["edos_definition"])}</p>')
    argument = html.escape(row["evidence"], quote=False)
    return (f'{context}<br><strong>Argument</strong><br>'
            f'<div style="white-space: pre-wrap">{argument}</div><br>{prompt[0]}')


def build_survey(template, name, items):
    survey = copy.deepcopy(template)
    original = {e["PrimaryAttribute"]: e for e in survey["SurveyElements"] if e["Element"] == "SQ"}
    if not {"QID7", "QID21", "QID29", "QID31", "QID4"} <= original.keys():
        raise ValueError("Draft must contain task, instruction, both ratings, and notes")
    for kind in ("BL", "FL", "QC", "SO"):
        if sum(e["Element"] == kind for e in survey["SurveyElements"]) != 1:
            raise ValueError(f"Expected exactly one {kind} element")
    draft_blocks = next(e["Payload"] for e in survey["SurveyElements"] if e["Element"] == "BL")

    def block_containing(qid):
        matches = [block for block in draft_blocks.values() if any(
            entry.get("QuestionID") == qid for entry in block["BlockElements"])]
        if len(matches) != 1:
            raise ValueError(f"Expected one draft block containing {qid}")
        return matches[0]

    task_block, item_block = block_containing("QID7"), block_containing("QID29")
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

    def add_block(draft, description, question_ids):
        index = len(blocks) + 1
        block = copy.deepcopy(draft)
        bid = f"BL_model_{index:04}"
        block.update(Type="Standard", Description=description, ID=bid,
                     BlockElements=[{"Type": "Question", "QuestionID": qid} for qid in question_ids])
        block.setdefault("Options", {})["RandomizeQuestions"] = "false"
        blocks[str(index - 1)] = block
        flow.append({"ID": bid, "Type": "Standard", "FlowID": f"FL_{index + 1}"})

    intro, _ = add_question("QID7", "Task_description")
    intro_ids = [intro]
    if name == "comune":
        qid, payload = add_question("QID29", "Annotator", "<p><strong>Annotatrice</strong></p><p>Seleziona il tuo nome.</p>")
        payload["Choices"] = {str(i): {"Display": annotator} for i, annotator in enumerate(ANNOTATORS, 1)}
        payload["ChoiceOrder"] = ["1", "2", "3"]
        intro_ids.append(qid)
    add_block(task_block, "Task description", intro_ids)
    for position, item in enumerate(items, 1):
        row, identity = item["row"], item["question_id"]
        instruction_id, _ = add_question("QID21", f"Instruction_{identity}")
        rating_ids = {}
        for dimension, template_id in (("faithfulness", "QID29"), ("answer_relevance", "QID31")):
            tag = f"{dimension}_{identity}"
            text = render_rating(row, original[template_id]["Payload"]["QuestionText"], dimension)
            qid, _ = add_question(template_id, tag, text)
            rating_ids[f"{dimension}_question_id"] = qid
            rating_ids[f"{dimension}_export_tag"] = tag
        notes_tag = f"Notes_{identity}"
        notes_id, _ = add_question("QID4", notes_tag)
        add_block(item_block, f"{row['primary_edos_label']} | {identity}", [
            instruction_id, rating_ids["faithfulness_question_id"],
            rating_ids["answer_relevance_question_id"], notes_id])
        mapping.append({
            "question_id": identity, "survey_id": name, "position": position,
            "card_id": row["card_id"], "generator_model": row["generator_model"],
            "primary_edos_label": row["primary_edos_label"], "edos_definition": row["edos_definition"],
            "reranker_rank": row["retrieval"]["reranker_rank"], "chunk_id": row["chunk"]["chunk_id"],
            "query_id": row["retrieval"]["query_id"], "source_file": item["source_file"],
            "source_id": row["source"]["source_id"], "markdown_file": row["source"]["markdown_file"],
            "instruction_question_id": instruction_id, **rating_ids,
            "notes_question_id": notes_id, "notes_export_tag": notes_tag,
        })
    title = f"model_annotation_pilot_{name}"
    survey["SurveyEntry"]["SurveyName"] = title
    survey["SurveyElements"] = [e for e in survey["SurveyElements"] if e["Element"] != "SQ"] + questions
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


def build_package(input_files=INPUT_FILES, template_path=TEMPLATE, seed=SEED):
    items, source_hashes = load_pool(input_files)
    pools = split_pool(items, seed)
    template = json.loads(Path(template_path).read_text(encoding="utf-8"))
    surveys, mappings = {}, {}
    for name, pool in pools.items():
        surveys[name], mappings[name] = build_survey(template, name, pool)
    ids = [row["question_id"] for rows in mappings.values() for row in rows]
    if len(ids) != N_ITEMS or len(set(ids)) != N_ITEMS:
        raise ValueError("Every model/card item must belong to exactly one survey")
    if [len(pools[name]) for name in pools] != [COMMON_SIZE, *[PRIVATE_SIZE] * 3]:
        raise ValueError("Incorrect common/private pool sizes")
    labels = sorted({item["row"]["primary_edos_label"] for item in items}, key=label_key)
    counts = {name: Counter(row["primary_edos_label"] for row in rows) for name, rows in mappings.items()}
    counts_by_label = {label: {name: counts[name][label] for name in pools} for label in labels}
    models = sorted({item["row"]["generator_model"] for item in items})
    counts_by_label_and_model = {
        label: {name: {model: sum(row["primary_edos_label"] == label and row["generator_model"] == model
                                 for row in rows) for model in models}
                for name, rows in mappings.items()} for label in labels
    }
    for label, label_counts in counts_by_label.items():
        private = [label_counts[name] for name in ANNOTATORS]
        if max(private) - min(private) > 1:
            raise ValueError(f"Private pools are not balanced within label {label}")
        common_models = sorted(counts_by_label_and_model[label]["comune"].values())
        if common_models != ([2, 3] if label_counts["comune"] == 5 else [2, 2]):
            raise ValueError(f"Common pool is not balanced by model within label {label}")
    manifest = {
        "seed": seed, "top_k": TOP_K, "n_models": 2, "n_labels": 11,
        "n_selections": N_SELECTIONS, "n_items": N_ITEMS,
        "item_identity": ["card_id", "generator_model"],
        "rating_dimensions": ["faithfulness", "answer_relevance"], "rating_scale": [0, 1, 2],
        "common_size": COMMON_SIZE, "private_size_per_annotator": PRIVATE_SIZE,
        "items_per_annotator": COMMON_SIZE + PRIVATE_SIZE,
        "rating_answers_per_annotator": 2 * (COMMON_SIZE + PRIVATE_SIZE),
        "total_expected_item_judgments": 3 * (COMMON_SIZE + PRIVATE_SIZE),
        "common_sampling": "Proportional allocation by label, largest remainders, seeded tie-breaks; two cards per model per label, plus one card from a seeded random model for the five-card label; seeded sampling within each label/model.",
        "private_sampling": "Disjoint; exactly equal total sizes; per-label counts differ by at most one.",
        "presentation": "Numeric EDOS order; sorted item IDs shuffled within label using random.Random(f'{seed}:presentation:{name}'); no Qualtrics randomizer.",
        "paragraph_display": "Remove the leading Section: marker, retaining its title/path and the body; preserve source HTML tables.",
        "input_files_sha256": source_hashes, "template_sha256": file_hash(Path(template_path)),
        "builder_sha256": file_hash(Path(__file__)), "counts_by_label": counts_by_label,
        "counts_by_label_and_model": counts_by_label_and_model,
        "counts_by_model": {name: dict(sorted(Counter(row["generator_model"] for row in rows).items()))
                            for name, rows in mappings.items()},
        "surveys": {name: {"qsf": f"{name}.qsf", "question_map": f"{name}_question_map.csv",
                           "n_items": len(rows), "annotators": list(ANNOTATORS) if name == "comune" else [name],
                           "items": rows} for name, rows in mappings.items()},
    }
    return surveys, mappings, manifest


def serialize_package(surveys, mappings, manifest):
    outputs = {}
    for name, survey in surveys.items():
        outputs[f"{name}.qsf"] = json.dumps(survey, ensure_ascii=False, indent=2) + "\n"
        stream = io.StringIO(newline="")
        writer = csv.DictWriter(stream, fieldnames=MAP_FIELDS)
        writer.writeheader()
        writer.writerows(mappings[name])
        outputs[f"{name}_question_map.csv"] = stream.getvalue()
    manifest = copy.deepcopy(manifest)
    manifest["output_files_sha256"] = {name: hashlib.sha256(content.encode("utf-8")).hexdigest()
                                      for name, content in outputs.items()}
    outputs["annotation_assignment.json"] = json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    return outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-files", type=Path, nargs=2, default=INPUT_FILES)
    parser.add_argument("--template", type=Path, default=TEMPLATE)
    parser.add_argument("--output-dir", type=Path, default=HERE)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--overwrite", action="store_true", help="Replace generated files if their contents differ")
    args = parser.parse_args()
    outputs = serialize_package(*build_package(args.input_files, args.template, args.seed))
    for name, content in outputs.items():
        path = args.output_dir / name
        if path.exists() and path.read_bytes() != content.encode("utf-8") and not args.overwrite:
            parser.error(f"{path} differs; use --overwrite to replace generated questionnaires")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, content in outputs.items():
        (args.output_dir / name).write_bytes(content.encode("utf-8"))
    for name in ("comune", *ANNOTATORS):
        size = COMMON_SIZE if name == "comune" else PRIVATE_SIZE
        print(f"{name}.qsf: {size} items, each with two ratings and optional notes")
    print(f"{N_ITEMS} model-specific cards; {COMMON_SIZE + PRIVATE_SIZE} items per annotator "
          f"({2 * (COMMON_SIZE + PRIVATE_SIZE)} rating answers). Seed: {args.seed}.")


if __name__ == "__main__":
    main()
