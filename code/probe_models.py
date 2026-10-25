"""Probe the configured LLM gateway: compliance, calibration, throughput.

Why this exists
---------------
Everything the mechanism pipeline does is LLM calls: the RL persuader's
message decoder, the crowd's internalization chain, the exposure accept
gate, and the reward-side estimators. A full 300-iteration arm costs
~1.8M unique calls, so choosing a model is a real decision — and it must
be made on *our* call shapes, not on generic benchmarks. Two things can
silently ruin a run:

- **non-compliance**: the accept gate is parsed with exact equality
  (`decision == "ACCEPT"`), the BT comparer needs a bare token, the judge
  a number;
- **degeneracy**: a judge that always answers 0 (or 10), or a gate that
  always REJECTs, flattens the reward with no error message.

Latency/throughput is the third axis, but the gateway is usually the
bottleneck, so measure it under the load you actually run at.

Usage
-----
    # both models: compliance + calibration + throughput at c=64
    PYTHONPATH=. .venv/bin/python code/probe_models.py

    # one model, production concurrency, no calibration block
    PYTHONPATH=. .venv/bin/python code/probe_models.py \
        --models "dynamo qwen38 27b/qwen38-27b-fp8" --concurrency 160

Notes
-----
- Inputs carry a per-run unique tag, so nothing is served from the cache
  (a repeated probe with fixed inputs silently measures the cache, not
  the model — an easy way to "discover" 2,700 calls/s).
- `reasoning_effort` comes from `.env` (`LLM_REASONING_EFFORT`). With it
  unset, a reasoning model spends the output budget on hidden reasoning
  and can return empty content; with `none` the same prompt answers in
  ~1/10 the time.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import statistics as st
import time

# absolute imports: this is a CLI tool, runnable as
# `PYTHONPATH=. python code/probe_models.py`
from code import LLM_module
from code.bt_judge import compare
from code.config import GLOBAL_LLM_CONFIG as C
from code.strategy import index_to_strategy

DEFAULT_MODELS = [
    "deepseek-dynamo/deepseek-v4.1-flash",
    "dynamo qwen38 27b/qwen38-27b-fp8",
]

_ACCEPT_PROBES = [
    "The levy will cost commuters 12% more while services get cut. Vote no.",
    "Every credible study agrees this policy works; opposition is misinformation.",
    "I lost my job because of this policy. Please think of people like me.",
    "As a transport engineer, the numbers simply do not support this plan.",
    "Everyone in our neighbourhood already switched. Join us and save money.",
    "The experts say it is safe, so stop worrying and accept the science.",
    "This policy protects our children's future; opposing it is reckless.",
    "The fee is a small price for cleaner air and faster buses.",
    "Politicians ignored us for years; this levy is more of the same.",
    "Data from three cities shows ridership rose 8% after a similar levy.",
]
_PROFILES = [
    "A 34-year-old data engineer who distrusts emotional appeals.",
    "A 58-year-old small-business owner worried about costs.",
    "A 22-year-old student who cares about climate policy.",
    "A 45-year-old nurse who values personal stories.",
    "A 67-year-old retired teacher who trusts expert consensus.",
]


def _compliance(model: str, tag: str) -> None:
    """One call per pipeline function: latency + parseability."""
    C.model = model
    rows = [
        ("ping", lambda i: LLM_module.call_llm(
            "Reply with the single word: ok", temperature=0.0,
            max_tokens=64, cache=False)),
        ("render_strategy", lambda i: LLM_module.render_strategy_message(
            index_to_strategy(0),
            target_profile=f"Engineer {tag}-{i}, 34, data-driven.",
            topic=f"transit levy {tag}-{i}", stance_direction="opposing")),
        ("accept_message", lambda i: LLM_module.accept_message(
            f"Policy {tag}-{i} is dangerous.", f"Engineer {tag}-{i}.", [])),
        ("generate_response", lambda i: LLM_module.generate_response_message(
            f"Policy {tag}-{i} is dangerous.", f"Engineer {tag}-{i}.", 0.3)),
        ("update_profile", lambda i: LLM_module.update_profile(
            received=f"Policy {tag}-{i} is dangerous.",
            profile=f"Engineer {tag}-{i}.")),
        ("generate_info", lambda i: LLM_module.generate_information(
            received=f"Policy {tag}-{i} is dangerous.",
            profile=f"Engineer {tag}-{i}.")),
        ("judge_persuasion", lambda i: LLM_module.judge_persuasiveness(
            f"Post {tag}-{i}: the levy hurts commuters.",
            f"Engineer {tag}-{i}.")),
        ("bt_compare", lambda i: compare(
            f"Post {tag}-{i} A: data shows the levy fails.",
            f"Post {tag}-{i} B: everyone agrees the levy is fine.",
            f"Engineer {tag}-{i}.", cache=False)),
    ]
    for label, fn in rows:
        lat, out, err = [], "", None
        for i in range(3):
            s = time.time()
            try:
                out = fn(i)
            except Exception as exc:  # noqa: BLE001 - probe must not die
                err = f"{type(exc).__name__}: {str(exc)[:60]}"
            lat.append(time.time() - s)
        lat.sort()
        flag = ""
        if label == "accept_message" and str(out) not in ("ACCEPT", "REJECT"):
            flag = "  <-- NOT EXACTLY ACCEPT/REJECT"
        if label == "bt_compare" and out not in (1, -1, 0):
            flag = "  <-- UNPARSEABLE"
        print(f"  {label:18s} {lat[len(lat) // 2]:5.2f}s  {str(out)[:66]!r}"
              + (f"  <-- {err}" if err else "") + flag)


def _calibration(model: str) -> None:
    """Is the accept gate and the reward-side judge non-degenerate?"""
    C.model = model
    judge, accept = [], []
    for i, post in enumerate(_ACCEPT_PROBES):
        prof = _PROFILES[i % len(_PROFILES)]
        judge.append(LLM_module.judge_persuasiveness(post, prof))
        accept.append(LLM_module.accept_message(post, prof, []))
    n_acc = sum(1 for a in accept if a == "ACCEPT")
    print(f"  judge: mean {st.mean(judge):.2f} sd {st.pstdev(judge):.2f} "
          f"distinct {len(set(judge))}/{len(judge)}"
          + ("  <-- DEGENERATE" if len(set(judge)) <= 2 else ""))
    print(f"  accept: {n_acc}/{len(accept)} ACCEPT"
          + ("  <-- DEGENERATE" if n_acc in (0, len(accept)) else ""))


def _throughput(model: str, conc: int, n: int, tag: str) -> None:
    """Sustained calls/s for the dominant (uncacheable) call shape."""
    C.model = model
    uni = f"{tag}-{conc}"

    def one(i: int) -> float | None:
        s = time.time()
        try:
            LLM_module.generate_information(
                received=f"M{uni}-{i}", profile=f"R{uni}-{i}")
            return time.time() - s
        except Exception:  # noqa: BLE001
            return None

    t = time.time()
    with cf.ThreadPoolExecutor(max_workers=conc) as pool:
        lat = list(pool.map(one, range(n)))
    wall = time.time() - t
    ok = [x for x in lat if x is not None]
    if not ok:
        print(f"  throughput @c={conc}: ALL FAILED")
        return
    print(f"  throughput @c={conc}: {len(ok)}/{n} ok, wall {wall:.1f}s, "
          f"mean {sum(ok) / len(ok):.2f}s -> {len(ok) / wall:.1f} calls/s")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", nargs="*", default=DEFAULT_MODELS)
    ap.add_argument("--concurrency", type=int, default=64,
                    help="production rollout worker count (default 64)")
    ap.add_argument("--throughput-n", type=int, default=64,
                    help="calls per throughput measurement")
    ap.add_argument("--no-calibration", action="store_true")
    ap.add_argument("--no-compliance", action="store_true")
    args = ap.parse_args()

    tag = str(int(time.time() * 1000))[-6:]  # unique: defeats the cache
    for model in args.models:
        print(f"\n===== {model} | reasoning_effort={C.reasoning_effort} "
              f"| max_tokens={C.generation_max_tokens} =====")
        if not args.no_compliance:
            _compliance(model, tag)
        if not args.no_calibration:
            _calibration(model)
        _throughput(model, args.concurrency, args.throughput_n, tag)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
