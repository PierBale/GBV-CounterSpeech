#!/usr/bin/env python3
"""
10_card_review_app.py

Small Streamlit interface to browse and present EDOS-specific quote-first cards,
validate each field, optionally revise field values, and export annotations.

Run from the repository root:

    pip install streamlit pandas
    streamlit run scripts/10_card_review_app.py

Optional defaults can be edited in the sidebar:
    data/cards/candidates/candidate_cards_normalized.jsonl
    data/validation/card_review_annotations.csv
    data/cards/validated/card_review_export.jsonl

This is intentionally a lightweight presentation/review tool, not a final
annotation platform.
"""

from __future__ import annotations

import json
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd
import streamlit as st


# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------

FIELD_ORDER = [
    "source",
    "chunk",
    "reasoning",
    "argument",
    "primary_edos_label",
    "secondary_edos_labels",
    "edos_alignment",
    "retrieval_keywords",
]

FIELD_VALIDATION_OPTIONS = ["OK", "Revise", "Reject", "Unsure"]

DEFAULT_CARDS_PATH = "data/cards/candidates/candidate_cards_normalized.jsonl"
DEFAULT_ANNOTATIONS_PATH = "data/validation/card_review_annotations.csv"
DEFAULT_EXPORT_PATH = "data/cards/validated/card_review_export.jsonl"

EDOS_LABELS = [
    "1.1 threats of harm",
    "1.2 incitement and encouragement of harm",
    "2.1 descriptive attacks",
    "2.2 aggressive and emotive attacks",
    "2.3 dehumanising attacks & overt sexual objectification",
    "3.1 casual use of gendered slurs, profanities, and insults",
    "3.2 immutable gender differences and gender stereotypes",
    "3.3 backhanded gendered compliments",
    "3.4 condescending explanations or unwelcome advice",
    "4.1 supporting mistreatment of individual women",
    "4.2 supporting systemic discrimination against women as a group",
]


# ---------------------------------------------------------------------
# File helpers
# ---------------------------------------------------------------------

def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Read JSONL cards from disk."""
    cards: List[Dict[str, Any]] = []
    if not path.exists():
        return cards
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                cards.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
    return cards


def read_jsonl_from_upload(uploaded_file: Any) -> List[Dict[str, Any]]:
    """Read JSONL cards from a Streamlit uploaded file."""
    content = uploaded_file.getvalue().decode("utf-8")
    cards: List[Dict[str, Any]] = []
    for line_no, line in enumerate(content.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            cards.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid uploaded JSONL at line {line_no}: {exc}") from exc
    return cards


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_annotations(path: Path) -> Dict[str, Dict[str, Any]]:
    """Load saved annotations keyed by card_id."""
    if not path.exists():
        return {}

    df = pd.read_csv(path, dtype=str).fillna("")
    if "card_id" not in df.columns:
        return {}

    annotations: Dict[str, Dict[str, Any]] = {}
    for _, row in df.iterrows():
        card_id = row["card_id"]
        annotations[card_id] = row.to_dict()
    return annotations


def save_annotation(path: Path, annotation: Dict[str, Any]) -> None:
    """Upsert a single annotation row into a CSV file."""
    path.parent.mkdir(parents=True, exist_ok=True)

    existing = load_annotations(path)
    existing[annotation["card_id"]] = annotation

    # Stable column order. Extra dynamic columns are appended.
    preferred = [
        "card_id",
        "overall_decision",
        "overall_notes",
        "reviewer",
        "review_timestamp",
        "faithfulness_score",
        "edos_specificity_score",
        "usefulness_score",
        "field_source_status",
        "field_source_comment",
        "field_source_revised",
        "field_chunk_status",
        "field_chunk_comment",
        "field_chunk_revised",
        "field_reasoning_status",
        "field_reasoning_comment",
        "field_reasoning_revised",
        "field_argument_status",
        "field_argument_comment",
        "field_argument_revised",
        "field_primary_edos_label_status",
        "field_primary_edos_label_comment",
        "field_primary_edos_label_revised",
        "field_secondary_edos_labels_status",
        "field_secondary_edos_labels_comment",
        "field_secondary_edos_labels_revised",
        "field_edos_alignment_status",
        "field_edos_alignment_comment",
        "field_edos_alignment_revised",
        "field_retrieval_keywords_status",
        "field_retrieval_keywords_comment",
        "field_retrieval_keywords_revised",
    ]

    all_keys = set()
    for item in existing.values():
        all_keys.update(item.keys())

    columns = preferred + sorted(k for k in all_keys if k not in preferred)

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for card_id in sorted(existing.keys()):
            row = {k: existing[card_id].get(k, "") for k in columns}
            writer.writerow(row)


# ---------------------------------------------------------------------
# Card normalization helpers
# ---------------------------------------------------------------------

def source_to_string(source: Any) -> str:
    if isinstance(source, dict):
        parts = []
        title = source.get("title", "")
        publisher = source.get("publisher", "")
        year = source.get("year", "")
        page = source.get("page", "")
        section = source.get("section", "")
        url = source.get("url", "")
        file_name = source.get("file_name", "")
        if title:
            parts.append(f"Title: {title}")
        if publisher:
            parts.append(f"Publisher: {publisher}")
        if year:
            parts.append(f"Year: {year}")
        if page not in [None, ""]:
            parts.append(f"Page: {page}")
        if section:
            parts.append(f"Section: {section}")
        if url:
            parts.append(f"URL: {url}")
        if file_name:
            parts.append(f"PDF file: {file_name}")
        return "\n".join(parts)
    return str(source)


def value_to_string(value: Any) -> str:
    if isinstance(value, dict):
        return source_to_string(value)
    if isinstance(value, list):
        return "; ".join(str(v) for v in value)
    if value is None:
        return ""
    return str(value)


def parse_semicolon_list(text: str) -> List[str]:
    return [x.strip() for x in text.split(";") if x.strip()]


def apply_annotation_to_card(card: Dict[str, Any], ann: Dict[str, Any]) -> Dict[str, Any]:
    """Create a validated/revised card preview using revised field values when present."""
    updated = json.loads(json.dumps(card, ensure_ascii=False))
    updated["review"] = {
        "overall_decision": ann.get("overall_decision", ""),
        "overall_notes": ann.get("overall_notes", ""),
        "faithfulness_score": ann.get("faithfulness_score", ""),
        "edos_specificity_score": ann.get("edos_specificity_score", ""),
        "usefulness_score": ann.get("usefulness_score", ""),
    }

    for field in FIELD_ORDER:
        revised_key = f"field_{field}_revised"
        revised = ann.get(revised_key, "")
        if revised:
            if field in ["secondary_edos_labels", "retrieval_keywords"]:
                updated[field] = parse_semicolon_list(revised)
            elif field == "source":
                # Keep source as free-form revised text in preview, without trying to parse it.
                updated[field] = revised
            else:
                updated[field] = revised

    if ann.get("overall_decision") == "Accept":
        updated["status"] = "validated"
    elif ann.get("overall_decision") == "Reject":
        updated["status"] = "rejected"
    elif ann.get("overall_decision") == "Revise":
        updated["status"] = "revised_candidate"
    else:
        updated["status"] = card.get("status", "candidate")

    return updated


# ---------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------

def inject_css() -> None:
    st.markdown(
        """
        <style>
        .card-box {
            border: 1px solid #e6e6e6;
            border-radius: 18px;
            padding: 22px;
            background: #ffffff;
            box-shadow: 0 4px 16px rgba(0,0,0,0.06);
            margin-bottom: 18px;
        }
        .source-box {
            border-left: 5px solid #d1d5db;
            padding: 14px 18px;
            background: #f9fafb;
            border-radius: 10px;
            margin-top: 12px;
            margin-bottom: 12px;
            font-size: 0.95rem;
        }
        .quote-box {
            border-left: 5px solid #64748b;
            padding: 14px 18px;
            background: #f8fafc;
            border-radius: 10px;
            font-style: italic;
            margin-top: 12px;
            margin-bottom: 12px;
        }
        .label-pill {
            display: inline-block;
            padding: 5px 10px;
            margin: 3px 3px 3px 0;
            border-radius: 999px;
            background: #eef2ff;
            color: #3730a3;
            font-size: 0.86rem;
            font-weight: 600;
        }
        .keyword-pill {
            display: inline-block;
            padding: 4px 9px;
            margin: 3px 3px 3px 0;
            border-radius: 999px;
            background: #f1f5f9;
            color: #334155;
            font-size: 0.82rem;
        }
        .field-review {
            border: 1px solid #eeeeee;
            border-radius: 14px;
            padding: 14px;
            background: #fcfcfc;
            margin-bottom: 12px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_card(card: Dict[str, Any]) -> None:
    source = card.get("source", {})
    source_title = source.get("title", "") if isinstance(source, dict) else ""
    publisher = source.get("publisher", "") if isinstance(source, dict) else ""
    year = source.get("year", "") if isinstance(source, dict) else ""
    page = source.get("page", "") if isinstance(source, dict) else ""
    section = source.get("section", "") if isinstance(source, dict) else ""
    url = source.get("url", "") if isinstance(source, dict) else ""
    file_name = source.get("file_name", "") if isinstance(source, dict) else ""

    primary = card.get("primary_edos_label", "")
    secondary = card.get("secondary_edos_labels", []) or []
    keywords = card.get("retrieval_keywords", []) or []

    st.markdown('<div class="card-box">', unsafe_allow_html=True)

    st.markdown(f"### {card.get('card_id', 'unknown card')}")
    st.markdown(f"**Status:** `{card.get('status', 'candidate')}`")

    if primary:
        st.markdown(f'<span class="label-pill">{primary}</span>', unsafe_allow_html=True)
    for lab in secondary:
        st.markdown(f'<span class="label-pill">secondary: {lab}</span>', unsafe_allow_html=True)

    st.markdown("#### Reasoning")
    st.markdown(card.get("reasoning", ""))

    st.markdown("#### Argument")
    st.markdown(card.get("argument") or card.get("claim", ""))

    st.markdown("#### Source chunk")
    st.markdown(
        f'<div class="quote-box">{card.get("chunk") or card.get("source_quote", "")}</div>',
        unsafe_allow_html=True,
    )

    st.markdown("#### EDOS alignment")
    st.markdown(card.get("edos_alignment", ""))

    if keywords:
        st.markdown("#### Retrieval keywords")
        pills = " ".join([f'<span class="keyword-pill">{kw}</span>' for kw in keywords])
        st.markdown(pills, unsafe_allow_html=True)

    st.markdown("#### Source")
    source_lines = []
    if source_title:
        source_lines.append(f"**{source_title}**")
    if publisher or year:
        source_lines.append(f"{publisher} ({year})")
    if page not in [None, ""]:
        source_lines.append(f"Page: {page}")
    if section:
        source_lines.append(f"Section: {section}")
    if url:
        source_lines.append(f"[Open source]({url})")
    if file_name:
        source_lines.append(f"PDF file: `{file_name}`")
    st.markdown('<div class="source-box">' + "<br>".join(source_lines) + "</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


def field_review_widget(
    card: Dict[str, Any],
    existing: Dict[str, Any],
    field: str,
) -> Tuple[str, str, str]:
    value = value_to_string(card.get(field, ""))
    status_key = f"field_{field}_status"
    comment_key = f"field_{field}_comment"
    revised_key = f"field_{field}_revised"

    st.markdown('<div class="field-review">', unsafe_allow_html=True)
    st.markdown(f"**{field}**")
    st.code(value or "(empty)", language="text")

    current_status = existing.get(status_key, "OK") or "OK"
    status = st.radio(
        "Field decision",
        FIELD_VALIDATION_OPTIONS,
        index=FIELD_VALIDATION_OPTIONS.index(current_status) if current_status in FIELD_VALIDATION_OPTIONS else 0,
        horizontal=True,
        key=status_key,
    )

    comment = st.text_area(
        "Comment",
        value=existing.get(comment_key, ""),
        key=comment_key,
        height=70,
    )

    revised = st.text_area(
        "Revised value, optional",
        value=existing.get(revised_key, ""),
        key=revised_key,
        height=90,
        help="Use this only if you want to suggest a corrected version of this field.",
    )
    st.markdown("</div>", unsafe_allow_html=True)
    return status, comment, revised


def annotation_for_current_card(
    card: Dict[str, Any],
    reviewer: str,
    existing: Dict[str, Any],
) -> Dict[str, Any]:
    import datetime as _dt

    ann: Dict[str, Any] = {
        "card_id": card.get("card_id", ""),
        "reviewer": reviewer,
        "review_timestamp": _dt.datetime.now().isoformat(timespec="seconds"),
        "overall_decision": st.session_state.get("overall_decision", existing.get("overall_decision", "Accept")),
        "overall_notes": st.session_state.get("overall_notes", existing.get("overall_notes", "")),
        "faithfulness_score": st.session_state.get("faithfulness_score", existing.get("faithfulness_score", "3")),
        "edos_specificity_score": st.session_state.get("edos_specificity_score", existing.get("edos_specificity_score", "3")),
        "usefulness_score": st.session_state.get("usefulness_score", existing.get("usefulness_score", "3")),
    }

    for field in FIELD_ORDER:
        ann[f"field_{field}_status"] = st.session_state.get(
            f"field_{field}_status",
            existing.get(f"field_{field}_status", "OK"),
        )
        ann[f"field_{field}_comment"] = st.session_state.get(
            f"field_{field}_comment",
            existing.get(f"field_{field}_comment", ""),
        )
        ann[f"field_{field}_revised"] = st.session_state.get(
            f"field_{field}_revised",
            existing.get(f"field_{field}_revised", ""),
        )
    return ann


def reset_field_session_state(card_id: str, existing: Dict[str, Any]) -> None:
    """Populate widgets with existing annotation when moving between cards."""
    st.session_state["current_card_id_for_widgets"] = card_id
    st.session_state["overall_decision"] = existing.get("overall_decision", "Accept") or "Accept"
    st.session_state["overall_notes"] = existing.get("overall_notes", "")
    st.session_state["faithfulness_score"] = int(existing.get("faithfulness_score", "3") or 3)
    st.session_state["edos_specificity_score"] = int(existing.get("edos_specificity_score", "3") or 3)
    st.session_state["usefulness_score"] = int(existing.get("usefulness_score", "3") or 3)

    for field in FIELD_ORDER:
        st.session_state[f"field_{field}_status"] = existing.get(f"field_{field}_status", "OK") or "OK"
        st.session_state[f"field_{field}_comment"] = existing.get(f"field_{field}_comment", "")
        st.session_state[f"field_{field}_revised"] = existing.get(f"field_{field}_revised", "")


# ---------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------

def main() -> None:
    st.set_page_config(
        page_title="EDOS Card Review",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_css()

    st.title("EDOS-specific Card Review")
    st.caption(
        "Lightweight presentation/review interface for quote-first cards. "
        "Use it to inspect each card, validate fields, and export a review file."
    )

    with st.sidebar:
        st.header("Input")
        cards_path_str = st.text_input("Cards JSONL path", DEFAULT_CARDS_PATH)
        uploaded = st.file_uploader("Or upload JSONL cards", type=["jsonl", "txt"])

        annotations_path_str = st.text_input("Annotations CSV path", DEFAULT_ANNOTATIONS_PATH)
        export_path_str = st.text_input("Reviewed JSONL export path", DEFAULT_EXPORT_PATH)

        reviewer = st.text_input("Reviewer name / initials", "")

        st.header("Filters")
        selected_label = st.selectbox("EDOS label", ["All"] + EDOS_LABELS)
        decision_filter = st.selectbox("Review status", ["All", "Reviewed", "Not reviewed"])

    cards_path = Path(cards_path_str)
    annotations_path = Path(annotations_path_str)
    export_path = Path(export_path_str)

    try:
        cards = read_jsonl_from_upload(uploaded) if uploaded is not None else read_jsonl(cards_path)
    except Exception as exc:
        st.error(f"Could not read cards: {exc}")
        return

    if not cards:
        st.warning("No cards loaded. Provide a JSONL path or upload a JSONL file.")
        return

    annotations = load_annotations(annotations_path)

    # Apply filters
    filtered_cards = []
    for card in cards:
        cid = card.get("card_id", "")
        if selected_label != "All" and card.get("primary_edos_label") != selected_label:
            continue
        reviewed = cid in annotations
        if decision_filter == "Reviewed" and not reviewed:
            continue
        if decision_filter == "Not reviewed" and reviewed:
            continue
        filtered_cards.append(card)

    if not filtered_cards:
        st.warning("No cards match the current filters.")
        return

    if "card_index" not in st.session_state:
        st.session_state["card_index"] = 0

    st.session_state["card_index"] = max(0, min(st.session_state["card_index"], len(filtered_cards) - 1))

    # Sidebar summary
    with st.sidebar:
        st.header("Progress")
        reviewed_count = sum(1 for c in cards if c.get("card_id") in annotations)
        st.write(f"Loaded cards: **{len(cards)}**")
        st.write(f"Reviewed cards: **{reviewed_count}**")
        st.write(f"Filtered cards: **{len(filtered_cards)}**")
        if len(cards) > 0:
            st.progress(reviewed_count / len(cards))

        st.header("Navigation")
        jump = st.number_input(
            "Card index",
            min_value=1,
            max_value=len(filtered_cards),
            value=st.session_state["card_index"] + 1,
            step=1,
        )
        if jump - 1 != st.session_state["card_index"]:
            st.session_state["card_index"] = jump - 1

    card = filtered_cards[st.session_state["card_index"]]
    card_id = card.get("card_id", "")
    existing = annotations.get(card_id, {})

    if st.session_state.get("current_card_id_for_widgets") != card_id:
        reset_field_session_state(card_id, existing)

    left, right = st.columns([1.1, 0.9], gap="large")

    with left:
        st.subheader(f"Card {st.session_state['card_index'] + 1} of {len(filtered_cards)}")
        render_card(card)

        nav1, nav2, nav3 = st.columns(3)
        with nav1:
            if st.button("← Previous", use_container_width=True, disabled=st.session_state["card_index"] == 0):
                st.session_state["card_index"] -= 1
                st.rerun()
        with nav2:
            if st.button("Save review", type="primary", use_container_width=True):
                ann = annotation_for_current_card(card, reviewer, existing)
                save_annotation(annotations_path, ann)
                st.success(f"Saved review for {card_id}")
                st.rerun()
        with nav3:
            if st.button("Next →", use_container_width=True, disabled=st.session_state["card_index"] >= len(filtered_cards) - 1):
                st.session_state["card_index"] += 1
                st.rerun()

    with right:
        st.subheader("Review")

        current_decision = existing.get("overall_decision", "Accept") or "Accept"
        decision_options = ["Accept", "Revise", "Reject", "Unsure"]
        st.radio(
            "Overall decision",
            decision_options,
            index=decision_options.index(current_decision) if current_decision in decision_options else 0,
            horizontal=True,
            key="overall_decision",
        )

        st.markdown("#### Overall scores")
        st.slider("Faithfulness to source quote", 1, 5, int(existing.get("faithfulness_score", "3") or 3), key="faithfulness_score")
        st.slider("EDOS specificity", 1, 5, int(existing.get("edos_specificity_score", "3") or 3), key="edos_specificity_score")
        st.slider("Usefulness for counterspeech", 1, 5, int(existing.get("usefulness_score", "3") or 3), key="usefulness_score")

        st.text_area(
            "Overall notes",
            value=existing.get("overall_notes", ""),
            height=100,
            key="overall_notes",
        )

        st.markdown("---")
        st.markdown("### Field-level validation")

        for field in FIELD_ORDER:
            field_review_widget(card, existing, field)

    st.markdown("---")
    st.subheader("Export")

    col_a, col_b, col_c = st.columns(3)

    with col_a:
        if st.button("Export reviewed JSONL", use_container_width=True):
            annotations = load_annotations(annotations_path)
            reviewed_cards = []
            for c in cards:
                cid = c.get("card_id", "")
                if cid in annotations:
                    reviewed_cards.append(apply_annotation_to_card(c, annotations[cid]))
            write_jsonl(export_path, reviewed_cards)
            st.success(f"Exported {len(reviewed_cards)} reviewed cards to {export_path}")

    with col_b:
        if annotations_path.exists():
            st.download_button(
                "Download annotations CSV",
                data=annotations_path.read_bytes(),
                file_name=annotations_path.name,
                mime="text/csv",
                use_container_width=True,
            )

    with col_c:
        if export_path.exists():
            st.download_button(
                "Download reviewed JSONL",
                data=export_path.read_bytes(),
                file_name=export_path.name,
                mime="application/jsonl",
                use_container_width=True,
            )

    with st.expander("Raw JSON for current card"):
        st.json(card)


if __name__ == "__main__":
    main()
