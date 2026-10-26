"""Build a Qualtrics survey with one rating question per block/page.

Examples (run from the repository root):
    python3 Qualtrics/build_retrieval_survey.py
    python3 Qualtrics/build_retrieval_survey.py --keep-common-copies

Select the top three reranker ranks per query in each JSONL file.
Chunks selected by at least two configurations for the same query appear once,
even at different ranks. Their identifier is "shared", or "all" when selected
by every configuration. Other chunks retain configuration-specific questions. --keep-common-copies
keeps every configuration's copy instead. Each exported question identifier
includes its chunk_id. Native Qualtrics QIDs are recorded separately in the
CSV, along with all source configurations and ranks for shared questions.
The task introduction stays first; Qualtrics randomizes the rating pages for
each respondent without changing question IDs or the CSV mapping order.
Only the Python standard library is needed.
"""

import argparse
import copy
import csv
import html
import json
import re
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
MAP_FIELDS = [
    'question_id', 'qualtrics_question_id', 'data_export_tag', 'pipeline_id', 'edos_label',
    'reranker_rank', 'chunk_id', 'query_id', 'document_source',
    'embedding_model', 'reranker_model',
    'run_id', 'query_text', 'source_matches',
]


def load_candidates(input_dir, top_n):
    """Return ranks 1..top_n for each query, independently of file order."""
    if not isinstance(top_n, int) or isinstance(top_n, bool) or top_n < 1:
        raise ValueError('top_n must be a positive integer')
    paths = sorted(input_dir.glob('*.jsonl'))
    if not paths:
        raise ValueError(f'No JSONL files found in {input_dir}')
    settings = {}
    required = {
        'label', 'definition', 'text', 'chunk_id', 'reranker_rank',
        'query_id', 'query_text', 'pipeline_id',
    }
    query_details = {}
    run_ids = set()
    for path in paths:
        queries = defaultdict(list)
        with path.open(encoding='utf-8') as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                missing = required - row.keys()
                if missing:
                    raise ValueError(f'{path}:{line_number}: missing {sorted(missing)}')
                if row.get('pipeline_id', path.stem) != path.stem:
                    raise ValueError(f'{path}:{line_number}: pipeline_id differs from filename')
                query_id = row['query_id']
                if not isinstance(query_id, str) or not query_id.strip():
                    raise ValueError(f'{path}:{line_number}: query_id must be nonempty')
                details = (row['label'], row['definition'], row['query_text'])
                if query_id in query_details and query_details[query_id] != details:
                    raise ValueError(f'{path}:{line_number}: conflicting query {query_id}')
                query_details[query_id] = details
                rank = row['reranker_rank']
                if not isinstance(rank, int) or isinstance(rank, bool) or rank < 1:
                    raise ValueError(f'{path}:{line_number}: reranker_rank must be a positive integer')
                if row.get('run_id'):
                    run_ids.add(row['run_id'])
                queries[query_id].append(row)
        if not queries:
            raise ValueError(f'No entries found in {path}')
        selected = {}
        for query_id, rows in queries.items():
            if len({row['reranker_rank'] for row in rows}) != len(rows):
                raise ValueError(f'{path.stem}: duplicate rank for {query_id}')
            if len({row['chunk_id'] for row in rows}) != len(rows):
                raise ValueError(f'{path.stem}: duplicate chunk_id for {query_id}')
            selected[query_id] = sorted(rows, key=lambda row: row['reranker_rank'])[:top_n]
        settings[path.stem] = (path, selected)

    if len(run_ids) > 1:
        raise ValueError('Input files contain different run_id values')
    all_queries = set.union(*(set(queries) for _, queries in settings.values()))
    for pipeline, (_, queries) in settings.items():
        for query_id in sorted(all_queries):
            rows = queries.get(query_id, [])
            if [row['reranker_rank'] for row in rows] != list(range(1, top_n + 1)):
                raise ValueError(f'{pipeline}: expected ranks 1..{top_n} for {query_id}')
    return settings


def common_chunks(settings):
    """Find chunks selected by at least two configurations for each query."""
    counts = defaultdict(lambda: defaultdict(set))
    for pipeline, (_, queries) in settings.items():
        for query_id, rows in queries.items():
            for row in rows:
                counts[query_id][row['chunk_id']].add(pipeline)
    return {
        query_id: {
            chunk_id for chunk_id, pipelines in chunks.items()
            if len(pipelines) >= 2
        }
        for query_id, chunks in counts.items()
    }


def render_question(row):
    """Preserve report HTML, including tables, and plain-text line breaks."""
    parts = re.split(r'(<table\b.*?</table>)', row['text'], flags=re.S | re.I)
    paragraph = ''.join(
        part if part.lower().startswith('<table') else
        '<div style="white-space: pre-wrap">' + part + '</div>'
        for part in parts if part
    )
    return (
        f'<p><strong>EDOS Label</strong></p><p><strong>{html.escape(row["label"])}</strong>'
        f' – {html.escape(row["definition"])}</p><br>'
        f'<p><strong>Paragraph</strong></p>{paragraph}<br>'
    )


def build_survey(template, settings, merge_common=True):
    survey = copy.deepcopy(template)
    elements = survey['SurveyElements']
    questions = {e['PrimaryAttribute']: e for e in elements if e['Element'] == 'SQ'}
    for qid in ('QID29', 'QID7'):
        if qid not in questions:
            raise ValueError(f'Template is missing {qid}')
    for kind in ('BL', 'FL', 'QC'):
        if sum(e['Element'] == kind for e in elements) != 1:
            raise ValueError(f'Template must contain exactly one {kind} element')
    next_id = max(int(qid[3:]) for qid in questions) + 1
    generated, mapping, flow = [], [], []
    blocks = {}

    def add_question(template_id, tag, text=None):
        nonlocal next_id
        element = copy.deepcopy(questions[template_id])
        qid = f'QID{next_id}'
        next_id += 1
        element.update(PrimaryAttribute=qid, SecondaryAttribute=tag)
        payload = element['Payload']
        payload.update(QuestionID=qid, DataExportTag=tag, QuestionDescription=tag)
        if text is not None:
            payload['QuestionText'] = text
        generated.append(element)
        return qid

    def add_block(description, qid):
        index = len(blocks)
        bid = f'BL_retrieval{index + 1:04}'
        blocks[str(index)] = {
            'Type': 'Standard', 'Description': description, 'ID': bid,
            'BlockElements': [{'Type': 'Question', 'QuestionID': qid}],
            'Options': {
                'BlockLocking': 'false', 'RandomizeQuestions': 'false',
                'BlockVisibility': 'Expanded',
            },
        }
        flow.append({'ID': bid, 'Type': 'Standard', 'FlowID': f'FL_{index + 2}'})

    add_block('Task description', add_question('QID7', 'Task_description'))
    shared = common_chunks(settings) if merge_common else {}
    query_labels = {
        query_id: rows[0]['label']
        for _, queries in settings.values()
        for query_id, rows in queries.items()
    }
    query_ids = sorted(query_labels, key=lambda query_id: (query_labels[query_id], query_id))
    # Assign IDs in the original order; randomize only the presentation flow.
    candidates = [
        (pipeline, path, rows[query_id][position])
        for query_id in query_ids
        for position in range(len(next(iter(settings.values()))[1][query_id]))
        for pipeline, (path, rows) in sorted(settings.items())
    ]
    sources = defaultdict(list)
    for pipeline, path, row in candidates:
        sources[row['query_id'], row['chunk_id']].append((pipeline, path, row))
    seen_shared = set()
    for number, (pipeline, path, row) in enumerate(candidates, 1):
        key = row['query_id'], row['chunk_id']
        is_shared = row['chunk_id'] in shared.get(row['query_id'], set())
        if is_shared and key in seen_shared:
            continue
        if is_shared:
            seen_shared.add(key)
        matches = sources[key] if is_shared else [(pipeline, path, row)]
        if any(
            (match['text'], match['definition'], match['query_text']) !=
            (row['text'], row['definition'], row['query_text'])
            for _, _, match in matches
        ):
            raise ValueError(f'Conflicting text or definition for shared chunk {key}')
        source_matches = [{
            'pipeline_id': name, 'reranker_rank': match['reranker_rank'],
            'source_file': str(source.resolve()),
            'embedding_model': match.get('embedding_model', ''),
            'reranker_model': match.get('reranker_model', ''),
        } for name, source, match in matches]
        if is_shared:
            identifier = 'all' if len(matches) == len(settings) else 'shared'
        else:
            identifier = pipeline
        tag = f'Q{number}-{identifier}-{row["chunk_id"]}'
        qid = add_question('QID29', tag, render_question(row))
        add_block(tag, qid)
        mapping.append({
            'question_id': tag, 'qualtrics_question_id': qid,
            'data_export_tag': tag, 'pipeline_id': identifier,
            'edos_label': row['label'],
            'reranker_rank': json.dumps({m['pipeline_id']: m['reranker_rank'] for m in source_matches})
                if is_shared else row['reranker_rank'],
            'chunk_id': row['chunk_id'], 'query_id': row.get('query_id', ''),
            'document_source': row.get('document_source', ''),
            'embedding_model': json.dumps(sorted({m['embedding_model'] for m in source_matches}))
                if is_shared else row.get('embedding_model', ''),
            'reranker_model': json.dumps(sorted({m['reranker_model'] for m in source_matches}))
                if is_shared else row.get('reranker_model', ''),
            'run_id': row.get('run_id', ''), 'query_text': row['query_text'],
            'source_matches': json.dumps(source_matches, ensure_ascii=False),
        })
    if not mapping:
        raise ValueError('No rating questions selected')

    # Keep the introduction fixed and present every rating block in random order.
    randomized_flow = [flow[0], {
        'Type': 'BlockRandomizer', 'FlowID': f'FL_{len(flow) + 2}',
        'SubSet': len(flow) - 1, 'EvenPresentation': False, 'Flow': flow[1:],
    }]
    survey['SurveyElements'] = [e for e in elements if e['Element'] != 'SQ'] + generated
    for element in survey['SurveyElements']:
        if element['Element'] == 'BL':
            element['Payload'] = blocks
        elif element['Element'] == 'FL':
            element['Payload'] = {
                'Flow': randomized_flow, 'Properties': {'Count': len(flow) + 2},
                'FlowID': 'FL_1', 'Type': 'Root',
            }
        elif element['Element'] == 'QC':
            element['SecondaryAttribute'] = str(len(generated))

    ids = {element['PrimaryAttribute'] for element in generated}
    references = [block['BlockElements'][0]['QuestionID'] for block in blocks.values()]
    if len(ids) != len(generated) or len(set(references)) != len(references) or set(references) != ids:
        raise ValueError('Invalid question IDs or block references')
    if len({item['data_export_tag'] for item in mapping}) != len(mapping):
        raise ValueError('Duplicate export tags')
    return survey, mapping, len(candidates) - len(mapping)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--input-dir', type=Path, default=ROOT / 'data/card_creation_comparison/runs/confronto-02/reranking')
    parser.add_argument('--template', type=Path, default=HERE / 'Draft-retrieval-setting.qsf')
    parser.add_argument('--top-n', type=int, default=3, help='Top N reranker ranks per query per file (default: 3)')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--keep-common-copies', dest='merge_common', action='store_false',
                      help='Keep separate copies of shared chunks for each configuration')
    mode.add_argument('--merge-common-chunks', '--exclude-common-chunks', dest='merge_common',
                      action='store_true', help='Keep chunks shared by at least two configurations once (default); --exclude-common-chunks is a legacy alias and no longer removes them')
    parser.set_defaults(merge_common=True)
    parser.add_argument('--output', type=Path, help='Output QSF path (default: Qualtrics/Best_retrieval_config.qsf, with _with_common_copies when keeping duplicates)')
    parser.add_argument('--csv-output', type=Path, help='Question map path (default: output filename with _question_map.csv suffix)')
    args = parser.parse_args()

    if args.top_n < 1:
        parser.error('--top-n must be a positive integer')
    suffix = '' if args.merge_common else '_with_common_copies'
    output = args.output or HERE / f'Best_retrieval_config{suffix}.qsf'
    csv_output = args.csv_output or output.with_name(output.stem + '_question_map.csv')
    if output.resolve() == args.template.resolve():
        parser.error('--output must differ from the template path')
    if csv_output.resolve() in {output.resolve(), args.template.resolve()}:
        parser.error('--csv-output must differ from the template and QSF output paths')
    try:
        template = json.loads(args.template.read_text(encoding='utf-8'))
        settings = load_candidates(args.input_dir, args.top_n)


        # Count occurrences of each (query_id, chunk_id) across configurations.
        chunk_counts = defaultdict(int)

        for configuration, (_, metadata) in settings.items():
            for query_id, rows in metadata.items():
                for row in rows:
                    chunk_counts[(query_id, row['chunk_id'])] += 1

        # Include chunks shared by two, three or all configurations.
        shared_chunks = {
            (query_id, chunk_id): count
            for (query_id, chunk_id), count in chunk_counts.items()
            if count >= 2
        }

        print("\nChunks selected in at least 2 configurations:")
        for (query_id, chunk_id), count in sorted(shared_chunks.items()):
            print(f"Query: {query_id}, Chunk: {chunk_id}, Configurations: {count}")

        print(f"Number of shared chunks: {len(shared_chunks)}")

        # Build the survey
        survey, mapping, merged_count = build_survey(template, settings, args.merge_common)
    except (OSError, ValueError, KeyError) as exc:
        parser.error(str(exc))


    output.parent.mkdir(parents=True, exist_ok=True)
    csv_output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(survey, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    with csv_output.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=MAP_FIELDS)
        writer.writeheader()
        writer.writerows(mapping)
    print(f'Created {len(mapping)} rating questions, randomized for each respondent, plus the task introduction first.')
    shared_count = sum(item['pipeline_id'] in {'all', 'shared'} for item in mapping)
    print(f'Configurations: {len(settings)}; shared questions: {shared_count}; duplicate copies merged: {merged_count}.')
    print(f'QSF: {output}')
    print(f'Question map: {csv_output}')


if __name__ == '__main__':
    main()
