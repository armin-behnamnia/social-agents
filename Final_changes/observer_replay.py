"""Offline observer replay (no LLM). Re-runs the particle filter on a saved
message_events.csv for a grid of (beta, mean_influence, noise) and reports Pearson r
and MAE between estimated and true final stance of non-seed targets.
Usage: python -m source.observer_replay path/to/message_events.csv
Uses accept/reject only; true stance is used only for scoring."""
import sys, itertools
import numpy as np, pandas as pd

def replay(df, beta, infl, noise, ratio=0.25, rate=1.0, n=500, seed=0):
    rng = np.random.default_rng(seed)
    df = df.sort_values('period', kind='stable')
    est, parts, seeds = {}, {}, {}
    for r in df.itertuples():
        if int(r.is_seed) == 1:
            continue
        t, s = r.target_agent, r.source_agent
        P = parts.setdefault(t, [rng.uniform(-1, 1, n), np.ones(n) / n])
        p, w = P
        acc = str(r.decision).upper() == 'ACCEPT'
        pa = 1 / (1 + np.exp(-beta * p))
        w = w * (pa if acc else 1 - pa); w = w / w.sum() if w.sum() > 0 else np.ones(n) / n
        if 1 / np.sum(w ** 2) < 0.5 * n:
            idx = np.searchsorted(np.cumsum(w) / w.sum(), (np.arange(n) + rng.random()) / n)
            idx = np.minimum(idx, n - 1); p, w = p[idx], np.ones(n) / n
        src = est.get(s, float(r.source_stance))  # seeds: true; others: estimate
        p = p + (1.0 if acc else ratio) * infl * rate * (src - p) + rng.normal(0, noise, n)
        p = np.clip(p, -1, 1); parts[t] = [p, w]; est[t] = float(np.average(p, weights=w))
    last = df[df.is_seed == 0].groupby('target_agent')['target_stance_after'].last()
    common = [a for a in est if a in last.index]
    e = np.array([est[a] for a in common]); t = last.loc[common].values
    r = np.corrcoef(e, t)[0, 1] if len(e) > 2 and e.std() > 0 else float('nan')
    return r, float(np.mean(np.abs(e - t))), len(common)

if __name__ == '__main__':
    path = sys.argv[1]
    df = pd.read_csv(path, sep=None, engine='python', encoding='utf-8-sig')
    print('beta infl noise | pearson_r MAE n')
    for b, i, nz in itertools.product([1, 3, 6], [0.1, 0.5, 1.0], [0.0, 0.02, 0.05]):
        r, m, n = replay(df, b, i, nz)
        print(f'{b} {i} {nz} | {r:.3f} {m:.3f} {n}')
