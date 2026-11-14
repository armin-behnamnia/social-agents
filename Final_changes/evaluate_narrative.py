"""Post-hoc public narrative analysis using actually transmitted messages.
Lexical distance is a proxy, NOT a factual distortion score.
Optional claim annotations are produced separately by a human/LLM reviewer.
"""
import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path


def distance(a, b):
    a, b = set(re.findall(r'\w+', a.lower())), set(re.findall(r'\w+', b.lower()))
    return 1 - len(a & b) / len(a | b) if a | b else 0.0


def analyze(messages, transmissions):
    messages = [dict(row, text=row.get('text') or row.get('generated_post', ''),
                     agent_id=row.get('agent_id') or row.get('target_agent') or row.get('source_agent'),
                     original_message_id=row.get('original_message_id') or row['message_id'])
                for row in messages if row.get('message_id')]
    lookup = {row['message_id']: row for row in messages}
    counts = Counter(row['message_id'] for row in transmissions)
    records = []
    def depth(mid, visited):
        if mid in visited:
            raise ValueError('Cycle in message lineage')
        node = lookup[mid]
        parent = node.get('parent_message_id', '')
        if not parent:
            return 0
        if parent not in lookup:
            raise ValueError('Missing parent: ' + parent)
        return 1 + depth(parent, visited | {mid})
    for mid, count in counts.items():
        if mid not in lookup:
            raise ValueError('Transmitted message not found: ' + mid)
        node = lookup[mid]
        root = node['original_message_id']
        if root not in lookup:
            raise ValueError('Missing root: ' + root)
        parent = node.get('parent_message_id') or mid
        records.append(dict(message_id=mid, root_message_id=root,
                            parent_message_id=node.get('parent_message_id', ''),
                            author=node['agent_id'], creation_period=node['period'],
                            generation_depth=depth(mid, set()), transmissions=count,
                            public_text=node['text'], root_text=lookup[root]['text'],
                            parent_text=lookup[parent]['text'],
                            root_lexical_drift=distance(node['text'], lookup[root]['text']),
                            parent_lexical_drift=distance(node['text'], lookup[parent]['text'])))
    weights = sum(counts.values())
    report = dict(unique_transmitted_messages=len(records), transmissions=weights,
                  unique_mean_root_drift=sum(r['root_lexical_drift'] for r in records) / len(records) if records else None,
                  transmission_weighted_root_drift=sum(r['root_lexical_drift'] * r['transmissions'] for r in records) / weights if weights else None,
                  note='Only transmitted public posts; lexical proxy, not factual fidelity. Match time, generation depth and transmission budget before comparing conditions.')
    return report, records


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('run_directory')
    args = parser.parse_args()
    path = Path(args.run_directory)
    def load(name):
        with (path / name).open(newline='') as handle:
            return list(csv.DictReader(handle))
    report, rows = analyze(load('message_events.csv'), load('transmission_events.csv'))
    (path / 'narrative_report.json').write_text(json.dumps(report, indent=2))
    # Annotation queue contains public text only. No fabricated semantic judgments.
    for row in rows:
        row.update(claims_preserved=None, claims_omitted=None, claims_contradicted=None,
                   claims_added=None, framing=None)
    (path / 'narrative_annotation_queue.json').write_text(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
