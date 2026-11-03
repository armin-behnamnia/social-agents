"""Offline prediction evaluation. Dependencies: numpy, pandas, scikit-learn.
Example: python source/evaluate_prediction.py data/evaluation --out prediction_report.json
No private stance columns are used as features.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def read_runs(root):
    runs = []
    for path in Path(root).rglob('manifest.json'):
        manifest = json.loads(path.read_text())
        if manifest['condition'] != 'reference':
            continue
        def load(name):
            p = path.parent / (name + '.csv')
            return pd.read_csv(p) if p.exists() and p.stat().st_size > 1 else pd.DataFrame()
        runs.append((manifest, load('candidate_snapshots'),
                     load('diffusion_events'), load('observable_preinteraction')))
    return runs


def labels(snapshots, events, end, horizon):
    """Start-of-period forecast: h=1 includes events in the current period.
    A negative is observed only when its full horizon has elapsed.
    """
    accepted = events[events.adopted.astype(str).str.lower().isin(['true', '1'])] if len(events) else pd.DataFrame()
    first = accepted.groupby('target_agent').period.min().to_dict() if len(accepted) else {}
    out = snapshots.copy()
    out['adoption_period'] = out.agent_id.map(first)
    out['label'] = ((out.adoption_period >= out.period) &
                    (out.adoption_period < out.period + horizon)).astype(int)
    out['censored'] = (out.label == 0) & (out.period + horizon > end)
    out['horizon'] = horizon
    return out


def score(frame, probability, k):
    frame = frame.copy()
    frame['probability'] = probability
    groups = []
    leads = []
    for _, group in frame.groupby(['run_id', 'period']):
        top = group.nlargest(min(k, len(group)), 'probability')
        positives = group.label.sum()
        groups.append((top.label.mean(), top.label.sum() / positives if positives else np.nan))
        hit = top[top.label == 1]
        leads.extend((hit.adoption_period - hit.period).tolist())
    return dict(pr_auc=float(average_precision_score(frame.label, probability)) if frame.label.sum() else None,
                brier=float(brier_score_loss(frame.label, probability)),
                precision_at_k=float(np.mean([x[0] for x in groups])),
                recall_at_k=float(np.nanmean([x[1] for x in groups])) if any(np.isfinite(x[1]) for x in groups) else None,
                lead_time_periods=float(np.mean(leads)) if leads else None)


def evaluate(data, out, k=10):
    groups = data.pair_id.astype(str)
    if groups.nunique() < 3:
        raise ValueError('At least three independent pair_id groups required; prefer many more for bootstrap.')
    train, test = next(GroupShuffleSplit(n_splits=1, test_size=.3, random_state=2026).split(data, groups=groups))
    numeric = ['degree_centrality', 'betweenness_centrality']
    if 'exposures_before' in data:
        numeric += ['exposures_before']
    report, predictions = {}, []
    for name, extra in [('without_stance', []), ('stance_aware', ['estimated_stance_before', 'estimated_alignment'] if 'estimated_alignment' in data else ['estimated_stance', 'stance_uncertainty'])]:
        features = ColumnTransformer([
            ('text', TfidfVectorizer(max_features=5000, min_df=1), 'profile_text'),
            ('numeric', make_pipeline(SimpleImputer(), StandardScaler()), numeric + extra)])
        model = make_pipeline(features, LogisticRegression(max_iter=1000, random_state=2026))
        if data.iloc[train].label.nunique() < 2:
            raise ValueError('Training labels need both classes.')
        model.fit(data.iloc[train], data.iloc[train].label)
        p = model.predict_proba(data.iloc[test])[:, 1]
        frame = data.iloc[test].copy()
        frame['probability'] = p
        frame['predictor'] = name
        predictions.append(frame)
        report[name] = score(frame, p, k)
    # Paired bootstrap resamples held-out pair groups, not individual events.
    rng = np.random.default_rng(2026)
    a, b = predictions
    ids = a.pair_id.unique()
    differences = []
    for _ in range(1000):
        sampled = rng.choice(ids, len(ids), replace=True)
        aa = pd.concat([a[a.pair_id == x] for x in sampled])
        bb = pd.concat([b[b.pair_id == x] for x in sampled])
        differences.append(brier_score_loss(bb.label, bb.probability) - brier_score_loss(aa.label, aa.probability))
    report['stance_minus_baseline_brier_ci95'] = np.quantile(differences, [.025, .975]).tolist()
    report['held_out_pair_ids'] = [str(x) for x in ids]
    report['bootstrap_warning'] = 'CI unreliable with few independent held-out groups; negative Brier difference favors stance.'
    Path(out).write_text(json.dumps(report, indent=2))
    pd.concat(predictions).to_csv(str(out) + '.predictions.csv', index=False)
    return report


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('root')
    parser.add_argument('--out', default='prediction_report.json')
    parser.add_argument('--horizon', type=int, default=1)
    parser.add_argument('--k', type=int, default=10)
    parser.add_argument('--task', choices=['network', 'exposure'], default='network')
    args = parser.parse_args()
    tables = []
    for manifest, snapshots, events, exposures in read_runs(args.root):
        if args.task == 'network':
            if snapshots.empty:
                continue
            table = labels(snapshots, events, manifest['observation_periods'], args.horizon)
            table = table[~table.censored]
        else:
            if exposures.empty or events.empty:
                continue
            table = exposures.merge(events[['event_id', 'adopted']], on='event_id', validate='one_to_one')
            table['label'] = table.adopted.astype(str).str.lower().isin(['true', '1']).astype(int)
            table['agent_id'] = table.target_agent
            table['adoption_period'] = np.where(table.label == 1, table.period, np.nan)
        tables.append(table)
    if not tables:
        raise ValueError('No eligible reference records.')
    evaluate(pd.concat(tables, ignore_index=True).fillna({'profile_text': ''}), args.out, args.k)


if __name__ == '__main__':
    main()
