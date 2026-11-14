"""Usage: python source/evaluate_runs.py data/evaluation --test-fraction 0.2

No model fitting: reports persistence/zero-change baselines and observer reconstruction.
Splits whole paired repetitions, NEVER individual interaction rows.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def score(folder):
    events = pd.read_csv(folder / 'diffusion_events.csv')
    features = pd.read_csv(folder / 'observable_preinteraction.csv')
    joined = events.merge(features[['event_id', 'predicted_stance_after_persistence']],
                          on='event_id', validate='one_to_one')
    result = {'interactions': len(joined)}
    for name, truth, pred in [
        ('early_persistence', joined.target_stance_after, joined.predicted_stance_after_persistence),
        ('zero_delta', joined.target_delta_stance, np.zeros(len(joined))),
        ('postinteraction_observer', joined.target_stance_after, joined.estimated_stance_after),
    ]:
        error = np.asarray(pred) - np.asarray(truth)
        result[name] = {'mae': float(np.abs(error).mean()),
                        'rmse': float(np.sqrt(np.mean(error ** 2)))}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--test-fraction', type=float, default=.2)
    args = parser.parse_args()
    if not 0 < args.test_fraction < 1:
        parser.error('test-fraction must be strictly between 0 and 1')
    folders = sorted(args.root.glob('*/*/manifest.json'))
    pairs = sorted({p.parent.parent.name for p in folders},
                   key=lambda x: hashlib.sha256(x.encode()).hexdigest())
    n_test = max(1, min(len(pairs) - 1, round(len(pairs) * args.test_fraction))) if len(pairs) > 1 else 0
    test_pairs = set(pairs[:n_test])
    results = []
    for manifest in folders:
        meta = json.loads(manifest.read_text(encoding='utf-8'))
        split = 'test' if meta['pair_id'] in test_pairs else 'train'
        try:
            metrics = score(manifest.parent)
        except pd.errors.EmptyDataError:
            metrics = {'interactions': 0}
        results.append(dict(pair_id=meta['pair_id'], condition=meta['condition'], split=split,
                            metrics=metrics))
    report = {'paired_repetitions': len(pairs),
              'warning': 'One pair cannot support held-out evaluation. No learned model was trained. '
                         'Do not use test outcomes for model selection.',
              'runs': results}
    args.root.mkdir(parents=True, exist_ok=True)
    output = args.root / 'evaluation_report.json'
    output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(output)

if __name__ == '__main__':
    main()
