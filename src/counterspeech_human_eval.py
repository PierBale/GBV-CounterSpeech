"""Excel files for the pairwise human evaluation of two systems (run from src after the generate stage).

  python counterspeech_human_eval.py [--output-dir ../data/counterspeech_v4] [--per-dataset 25]

For each comparison in COMPARISONS writes to <output dir>/human_eval/:
  <name>.xlsx      the file for the annotators, one sheet per dataset (EDOS, MultiCONAN; ids E01...,
                   M01...): id, hate speech, CS1, CS2, the question "Which is the best? A, B, Tie"
                   (A = CS1, B = CS2), and three comment columns
  <name>_key.csv   the key, to keep locally (not for the annotators): for every id the dataset,
                   the message id and the system shown as CS1 and as CS2

The same messages for every comparison: per-dataset random sample (seed 42) of the experiment
subset. Rows are shuffled within each dataset; which system is CS1 is random per row, balanced
within each dataset (half of the rows each, one more for one system when the count is odd). A comparison whose generations are missing is skipped.
"""

import argparse
import csv
import dataclasses
import random
from collections import defaultdict
from pathlib import Path

from unito_amazon.card_creation.io import read_jsonl
from unito_amazon.counterspeech_experiment.config import ExperimentConfig

COMPARISONS = {
    "rag_vs_card_rag": ("rag", "card_rag"),
    "card_rag_label_vs_rag_label_query": ("card_rag_label", "rag_label_query"),
}
DATASETS = ("edos", "multiconan")
SHEETS = {"edos": "EDOS", "multiconan": "MultiCONAN"}  # one sheet per dataset; ids start with its initial
SEED = 42
QUESTION = "Which is the best? A, B, Tie"
HEADERS = ("id", "hate speech", "CS1", "CS2", QUESTION, "commento CS1", "commento CS2", "commento generale")
WIDTHS = (8, 45, 55, 55, 16, 30, 30, 30)
FONT = "Arial"


def sample_messages(generations: dict, per_dataset: int) -> list[tuple[str, str]]:
    """(dataset, message id): the same messages for every comparison."""
    chosen = []
    for source in DATASETS:
        ids = sorted(message_id for (src, message_id) in generations if src == source)
        if len(ids) < per_dataset:
            raise ValueError(f"Only {len(ids)} messages for {source}.")
        chosen += [(source, message_id) for message_id in random.Random(f"{SEED}|{source}").sample(ids, per_dataset)]
    return chosen


def build_rows(name: str, systems: tuple[str, str], messages: list, generations: dict) -> list[dict]:
    """Per dataset: shuffled rows, ids E01... / M01..., and each system shown as CS1 in half of the
    rows (one more for one of the two when the count is odd), at random."""
    rows = []
    for source in DATASETS:
        rng = random.Random(f"{SEED}|{name}|{source}")
        order = [message for message in messages if message[0] == source]
        rng.shuffle(order)
        flips = [i % 2 == 1 for i in range(len(order))]
        rng.shuffle(flips)
        for number, ((_, message_id), flip) in enumerate(zip(order, flips), start=1):
            records = generations[(source, message_id)]
            first, second = (systems[1], systems[0]) if flip else systems
            rows.append({"id": f"{SHEETS[source][0]}{number:02d}", "dataset": source, "message_id": message_id,
                         "hate speech": " ".join(records[first]["hateful_message"].split()),
                         "CS1": records[first]["generated_counter_speech"].strip(), "CS1 system": first,
                         "CS2": records[second]["generated_counter_speech"].strip(), "CS2 system": second})
    return rows


def write_workbook(path: Path, rows: list[dict]) -> None:
    from openpyxl import Workbook
    from openpyxl.comments import Comment
    from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
    from openpyxl.worksheet.datavalidation import DataValidation

    header_font = Font(name=FONT, bold=True, color="FFFFFF")
    header_fill = PatternFill("solid", fgColor="2F4F6F")
    body_font = Font(name=FONT, size=10)
    input_fill = PatternFill("solid", fgColor="FFF6CC")  # cells the annotators fill in
    border = Border(*(Side(style="thin", color="D0D0D0"),) * 4)
    wrap = Alignment(wrap_text=True, vertical="top")

    workbook = Workbook()
    workbook.remove(workbook.active)
    for source in DATASETS:
        sheet_rows = [row for row in rows if row["dataset"] == source]
        if not sheet_rows:
            continue
        sheet = workbook.create_sheet(SHEETS[source])
        sheet.append(HEADERS)
        for column, width in enumerate(WIDTHS, start=1):
            cell = sheet.cell(row=1, column=column)
            cell.font, cell.fill, cell.border = header_font, header_fill, border
            cell.alignment = Alignment(wrap_text=True, vertical="center", horizontal="center")
            sheet.column_dimensions[cell.column_letter].width = width
        sheet.cell(row=1, column=5).comment = Comment("A = CS1, B = CS2, Tie = equally good. Choose from the list.",
                                                      "instructions")
        for row in sheet_rows:
            sheet.append([row["id"], row["hate speech"], row["CS1"], row["CS2"], None, None, None, None])
        for row_cells in sheet.iter_rows(min_row=2, max_row=len(sheet_rows) + 1):
            for cell in row_cells:
                cell.font, cell.border, cell.alignment = body_font, border, wrap
                if cell.column >= 5:
                    cell.fill = input_fill
            row_cells[4].alignment = Alignment(horizontal="center", vertical="top")
        choice = DataValidation(type="list", formula1='"A,B,Tie"', allow_blank=True, showErrorMessage=True,
                                errorTitle="Invalid answer", error="Choose A (CS1), B (CS2) or Tie.")
        sheet.add_data_validation(choice)
        choice.add(f"E2:E{len(sheet_rows) + 1}")
        sheet.freeze_panes = "B2"
        sheet.row_dimensions[1].height = 32
    workbook.save(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--output-dir", type=Path, help="experiment folder (default: data/counterspeech_v4)")
    parser.add_argument("--per-dataset", type=int, default=25)
    args = parser.parse_args()
    config = ExperimentConfig()
    if args.output_dir:
        config = dataclasses.replace(config, output_dir=args.output_dir.resolve())
    generations = defaultdict(dict)
    for record in read_jsonl(config.generations_path):
        generations[(record["source"], record["id"])][record["generation_strategy"]] = record
    messages = sample_messages(generations, args.per_dataset)
    out = config.output_dir / "human_eval"
    out.mkdir(parents=True, exist_ok=True)
    for name, systems in COMPARISONS.items():
        missing = {s for s in systems for key in messages if s not in generations[key]}
        if missing:
            print(f"{name}: skipped, no generations for {sorted(missing)} yet.")
            continue
        rows = build_rows(name, systems, messages, generations)
        write_workbook(out / f"{name}.xlsx", rows)
        with (out / f"{name}_key.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["id", "dataset", "message_id", "CS1 system", "CS2 system"])
            writer.writeheader()
            writer.writerows({key: row[key] for key in writer.fieldnames} for row in rows)
        print(f"{name}: {len(rows)} rows -> {out / (name + '.xlsx')} (key: {name}_key.csv)")


if __name__ == "__main__":
    main()
