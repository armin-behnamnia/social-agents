"""Live monitoring of a training run: exact values, descriptions, charts.

What it shows
-------------
Three layers, all from the same registry (`code/metrics_registry.py`), so
the numbers on screen and their descriptions can never drift:

1. **Iteration metrics** — the training curves from
   ``<run>/training_log.json`` (reward, stance, entropy, KL, losses,
   judge statistics, paired evaluations).
2. **Immediate variables** — from the ``trace/`` JSONL streams written by
   ``--trace``: every persuasion decision's persuasiveness / goal
   alignment / reward, every period's population state, every diffusion
   event's stance delta and influence probability. If a run was not
   traced, the dashboard says so instead of inventing numbers.
3. **Registry table** — every recorded variable with its unit and
   description, plus its exact current / min / max / mean.

Nothing is computed once and cached in a way that can lie: the JSONL
streams are *tailed* incrementally and every aggregate is recomputed
exactly from the records (no sampling), so the dashboard shows exact
values while the run is still writing.

Usage
-----
    # auto-detect the newest run under output/ and serve it
    PYTHONPATH=. .venv/bin/python code/live_monitor.py --port 3010

    # pin a run
    PYTHONPATH=. .venv/bin/python code/live_monitor.py \\
        output/ppo_mech_cascade/live --port 3010

    # terminal view / machine-readable exact values / static snapshot
    PYTHONPATH=. .venv/bin/python code/live_monitor.py --text
    PYTHONPATH=. .venv/bin/python code/live_monitor.py --json > state.json
    PYTHONPATH=. .venv/bin/python code/live_monitor.py --snapshot dash.html

Endpoints (server mode): ``/`` full page, ``/partial`` just the data body
(the page swaps this in every refresh, preserving scroll position),
``/api/state`` all exact values as JSON, ``/metrics.txt`` plain text,
``/healthz`` liveness.
"""

from __future__ import annotations

import argparse
import codecs
import html
import json
import threading
import time
from collections import Counter, deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from code.metrics_registry import (
    HISTOGRAMS,
    ITER_AGGREGATE,
    STREAMS,
    Field,
    chart_fields,
    field_map,
    flatten_record,
)
from code.run_trace import read_meta

TRACE_STREAMS = ("steps", "periods", "events", "episodes")

PALETTE = (
    "#2f6f9f", "#c0504d", "#4f8a3d", "#8a6d3b", "#7a5195",
    "#bc5090", "#ef5675", "#ffa600", "#00695c", "#5d4037",
)
INK = "#1a1a1a"
MUTED = "#6b6b6b"
GRID = "#e3e3e3"


# ----------------------------------------------------------------------
# formatting
# ----------------------------------------------------------------------
def fmt(v, digits: int = 6) -> str:
    """Exact-ish number formatting: full precision by default."""
    if v is None:
        return "—"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, str):
        return v
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        if v != v:  # NaN
            return "NaN"
        if v in (float("inf"), float("-inf")):
            return "∞" if v > 0 else "-∞"
        if v == 0:
            return "0"
        av = abs(v)
        if av >= 1e5 or av < 1e-4:
            return f"{v:.{digits}g}"
        return f"{v:.{digits}g}"
    return str(v)


def fmt_dur(seconds) -> str:
    if seconds is None or seconds != seconds:
        return "—"
    seconds = int(max(0, seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m:02d}m"
    if m:
        return f"{m}m {s:02d}s"
    return f"{s}s"


def fmt_age(seconds) -> str:
    if seconds is None:
        return "—"
    if seconds < 90:
        return f"{seconds:.0f}s ago"
    return f"{fmt_dur(seconds)} ago"


class Numeric:
    """Exact running statistics for one numeric variable."""

    __slots__ = ("n", "sum", "min", "max", "last")

    def __init__(self) -> None:
        self.n = 0
        self.sum = 0.0
        self.min = float("inf")
        self.max = float("-inf")
        self.last = None

    def add(self, v: float) -> None:
        self.n += 1
        self.sum += v
        self.min = min(self.min, v)
        self.max = max(self.max, v)
        self.last = v

    @property
    def mean(self):
        return self.sum / self.n if self.n else None

    def as_dict(self) -> dict:
        return {
            "n": self.n,
            "mean": self.mean,
            "min": None if self.n == 0 else self.min,
            "max": None if self.n == 0 else self.max,
            "last": self.last,
        }


# ----------------------------------------------------------------------
# incremental trace tailing
# ----------------------------------------------------------------------
class TraceTail:
    """Tail one JSONL stream, maintaining exact aggregates.

    Starts by reading whatever exists (so the aggregates cover the whole
    run, not just since the monitor opened) and then reads only new bytes
    on each poll. Aggregates are exact sums/counts/min/max plus fixed-bin
    histograms and per-iteration means — nothing is sampled.
    """

    def __init__(self, path: str | Path, stream: str, keep_recent: int = 40):
        self.path = Path(path)
        self.stream = stream
        self.fields: dict[str, Field] = field_map(stream)
        self.hist_fields = {f.name: f for f in HISTOGRAMS.get(stream, ())}
        self.iter_names = tuple(ITER_AGGREGATE.get(stream, ()))
        self.keep_recent = keep_recent
        self.reset()

    def reset(self) -> None:
        self.offset = 0
        self.buf = ""
        self._decoder = codecs.getincrementaldecoder("utf-8")("replace")
        self.records = 0
        self.numeric: dict[str, Numeric] = {}
        self.cats: dict[str, Counter] = {}
        self.hists: dict[str, list[int]] = {}
        self.per_iter: dict[int, dict[str, list[float]]] = {}
        self.per_period: dict[int, dict[str, list[float]]] = {}
        self.by_strategy: dict[str, dict] = {}
        self.recent: deque = deque(maxlen=self.keep_recent)
        self.first_ts = None
        self.last_ts = None
        self.bad_lines = 0

    # -- polling ------------------------------------------------------
    def poll(self) -> int:
        """Read new complete lines; returns how many records were added."""
        if not self.path.is_file():
            return 0
        with self.path.open("rb") as fh:
            fh.seek(self.offset)
            chunk = fh.read()
        self.offset += len(chunk)
        if not chunk and not self.buf:
            return 0
        text = self.buf + self._decoder.decode(chunk)
        lines = text.split("\n")
        self.buf = lines.pop()
        added = 0
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                self.bad_lines += 1
                continue
            self._add(record)
            added += 1
        return added

    def _add(self, rec: dict) -> None:
        self.records += 1
        ts = rec.get("ts")
        if isinstance(ts, (int, float)):
            if self.first_ts is None:
                self.first_ts = ts
            self.last_ts = ts
        iteration = rec.get("iteration")
        iteration = int(iteration) if isinstance(iteration, (int, float)) else None
        period = rec.get("period")
        period = int(period) if isinstance(period, (int, float)) else None

        for name, field in self.fields.items():
            if name not in rec:
                continue
            value = rec[name]
            if field.kind in ("num", "flag"):
                if isinstance(value, bool):
                    num = 1.0 if value else 0.0
                elif isinstance(value, (int, float)):
                    num = float(value)
                else:
                    num = None
                if num is None:
                    continue
                self.numeric.setdefault(name, Numeric()).add(num)
                if field.iter_agg and iteration is not None:
                    self.per_iter.setdefault(iteration, {}).setdefault(
                        name, [0.0, 0.0]
                    )[0] += num
                    self.per_iter[iteration][name][1] += 1.0
                if field.hist is not None and name in self.hist_fields:
                    lo, hi, bins = field.hist
                    self.hists.setdefault(name, [0] * bins)
                    self._bucket(name, num, lo, hi, bins)
            elif field.kind == "cat" and isinstance(value, str):
                self.cats.setdefault(name, Counter())[value] += 1

        # stream-specific extras
        if self.stream == "periods" and period is not None:
            slot = self.per_period.setdefault(period, {})
            for key in ("mean_stance", "n_adopted", "n_accept",
                        "n_persuader_events", "n_events"):
                v = rec.get(key)
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    acc = slot.setdefault(key, [0.0, 0.0])
                    acc[0] += float(v)
                    acc[1] += 1.0
        if self.stream == "steps":
            spec = rec.get("strategy")
            if isinstance(spec, str):
                slot = self.by_strategy.setdefault(spec, {
                    "count": 0, "reward_sum": 0.0, "reward_n": 0,
                    "pers_sum": 0.0, "pers_n": 0.0,
                    "strategy_index": rec.get("strategy_index"),
                })
                slot["count"] += 1
                rew = rec.get("reward")
                if isinstance(rew, (int, float)) and not isinstance(rew, bool):
                    slot["reward_sum"] += float(rew)
                    slot["reward_n"] += 1
                p = rec.get("persuasiveness")
                if isinstance(p, (int, float)) and not isinstance(p, bool):
                    slot["pers_sum"] += float(p)
                    slot["pers_n"] += 1
        self.recent.append(rec)

    def _bucket(self, name: str, value: float, lo: float, hi: float,
                bins: int) -> None:
        counts = self.hists[name]
        if value <= lo:
            counts[0] += 1
            return
        if value >= hi:
            counts[-1] += 1
            return
        idx = int((value - lo) / (hi - lo) * bins)
        counts[min(bins - 1, max(0, idx))] += 1

    # -- accessors ----------------------------------------------------
    @property
    def exists(self) -> bool:
        return self.path.is_file()

    def iter_series(self, name: str) -> list[tuple[float, float]]:
        """Per-iteration MEAN of an immediate variable, exact."""
        out = []
        for it in sorted(self.per_iter):
            acc = self.per_iter[it].get(name)
            if acc and acc[1]:
                out.append((float(it), acc[0] / acc[1]))
        return out

    def period_series(self, name: str) -> list[tuple[float, float]]:
        """Mean of a per-period variable by period number (within-episode)."""
        out = []
        for period in sorted(self.per_period):
            acc = self.per_period[period].get(name)
            if acc and acc[1]:
                out.append((float(period), acc[0] / acc[1]))
        return out

    def stats(self, name: str) -> dict:
        n = self.numeric.get(name)
        return n.as_dict() if n else {
            "n": 0, "mean": None, "min": None, "max": None, "last": None
        }


# ----------------------------------------------------------------------
# run state
# ----------------------------------------------------------------------
class RunState:
    """Everything known about one run directory, refreshed on demand."""

    def __init__(self, run_dir: str | Path, keep_recent: int = 40,
                 read_history_bytes: bool = True):
        self.dir = Path(run_dir)
        self.read_history_bytes = read_history_bytes
        self.history: list[dict] = []
        self.config: dict = {}
        self.strategy_value: dict = {}
        self.results: dict = {}
        self.meta: dict = {}
        self.tails: dict[str, TraceTail] = {}
        self.log_mtime = None
        self._initialised = False

    # -- loading ------------------------------------------------------
    def init_tails(self, keep_recent: int = 40) -> None:
        trace_dir = self.dir / "trace"
        self.meta = read_meta(trace_dir)
        self.tails = {
            stream: TraceTail(trace_dir / f"{stream}.jsonl", stream,
                              keep_recent=keep_recent)
            for stream in TRACE_STREAMS
        }
        self._initialised = True

    def poll(self) -> None:
        if not self._initialised:
            self.init_tails()
        log = self.dir / "training_log.json"
        if log.is_file():
            try:
                payload = json.loads(log.read_text())
                self.history = payload.get("history", []) or []
                self.config = payload.get("config", {}) or {}
                self.log_mtime = log.stat().st_mtime
            except (json.JSONDecodeError, OSError):
                pass
        sv = self.dir / "strategy_value.json"
        if sv.is_file():
            try:
                self.strategy_value = json.loads(sv.read_text())
            except (json.JSONDecodeError, OSError):
                pass
        res = self.dir / "results.json"
        if res.is_file():
            try:
                self.results = json.loads(res.read_text())
            except (json.JSONDecodeError, OSError):
                pass
        for tail in self.tails.values():
            tail.poll()

    # -- derived ------------------------------------------------------
    @property
    def last_record(self) -> dict:
        return self.history[-1] if self.history else {}

    def status(self) -> dict:
        cfg = self.config
        rec = self.last_record
        flat = flatten_record(rec) if rec else {}
        history = self.history
        iterations_total = cfg.get("iterations")
        elapsed = flat.get("elapsed_s")
        iterations_done = int(rec.get("iteration", 0)) if rec else 0
        iters_per_min = None
        eta = None
        if history and elapsed and iterations_done > 1:
            # rate over the whole curve (robust): total iters / total time
            iters_per_min = iterations_done / (float(elapsed) / 60.0)
            if iterations_total and iters_per_min > 0:
                remaining = max(0, int(iterations_total) - iterations_done)
                eta = remaining / iters_per_min * 60.0
        age = None
        if self.log_mtime:
            age = max(0.0, time.time() - self.log_mtime)
        traced = [s for s, t in self.tails.items() if t.exists and t.records]
        return {
            "run_dir": str(self.dir),
            "mechanism": flat.get("mechanism") or self.meta.get("mechanism"),
            "iteration": iterations_done,
            "iterations_total": iterations_total,
            "elapsed_s": elapsed,
            "iters_per_min": iters_per_min,
            "eta_s": eta,
            "updated_age_s": age,
            "trace_streams": traced,
            "trace_episodes": cfg.get("trace_episodes", 0),
            "model": flat.get("llm_model"),
            "workers": cfg.get("rollout_workers"),
            "episodes_per_iteration": cfg.get("episodes_per_iteration"),
            "agent_num": cfg.get("agent_num"),
            "n_persuaders": cfg.get("n_persuaders"),
            "topology": cfg.get("topology"),
            "seed": cfg.get("seed"),
            "n_periods": cfg.get("n_periods"),
            "obs_mode": None if cfg.get("obs_mode") is None else cfg.get("obs_mode"),
            "target_selection": cfg.get("target_selection"),
            "judge": cfg.get("judge"),
            "paired_eval_every": cfg.get("paired_eval_every"),
        }

    def series(self, name: str) -> list[tuple[float, float]]:
        out = []
        for rec in self.history:
            value = flatten_record(rec).get(name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            out.append((float(rec.get("iteration", len(out) + 1)), float(value)))
        return out

    def history_stats(self, name: str) -> dict:
        values = [v for _, v in self.series(name)]
        if not values:
            return {"n": 0, "mean": None, "min": None, "max": None, "last": None}
        return {
            "n": len(values),
            "mean": sum(values) / len(values),
            "min": min(values),
            "max": max(values),
            "last": values[-1],
        }

    def last_value(self, name: str):
        for rec in reversed(self.history):
            value = flatten_record(rec).get(name)
            if value is not None:
                return value
        return None


def _trace_stats(rs: RunState, stream: str, name: str) -> dict:
    tail = rs.tails.get(stream)
    if tail is None:
        return {"n": 0, "mean": None, "min": None, "max": None, "last": None}
    return tail.stats(name)


# ----------------------------------------------------------------------
# SVG chart primitives
# ----------------------------------------------------------------------
def _ticks(lo: float, hi: float, n: int = 5) -> list[float]:
    if hi <= lo:
        return [lo]
    step = (hi - lo) / max(1, n - 1)
    return [lo + step * i for i in range(n)]


def svg_line(
    series: list[dict],
    *,
    w: int = 560,
    h: int = 190,
    title: str = "",
    x_label: str = "iteration",
    y_digits: int = 4,
    pad_l: int = 64,
    pad_r: int = 16,
    pad_t: int = 34,
    pad_b: int = 32,
) -> str:
    """Multi-series line chart as inline SVG (no JS, no CDN)."""
    usable = [s for s in series if s.get("pts")]
    head = (
        f'<text x="8" y="16" font-size="12.5" font-weight="600" '
        f'fill="{INK}">{html.escape(title)}</text>'
    )
    if not usable:
        return (
            f'<svg viewBox="0 0 {w} {h}" width="100%" height="{h}" '
            f'role="img" aria-label="{html.escape(title)}">'
            f'{head}<text x="8" y="{h // 2}" font-size="12" fill="{MUTED}">'
            f"no data yet</text></svg>"
        )
    xs = [x for s in usable for x, _ in s["pts"]]
    ys = [y for s in usable for _, y in s["pts"]]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    if x1 == x0:
        x0, x1 = x0 - 0.5, x1 + 0.5
    if y1 == y0:
        delta = abs(y1) * 0.05 or 0.5
        y0, y1 = y0 - delta, y1 + delta
    else:
        span = y1 - y0
        y0 -= span * 0.06
        y1 += span * 0.06

    def sx(x):
        return pad_l + (x - x0) / (x1 - x0) * (w - pad_l - pad_r)

    def sy(y):
        return h - pad_b - (y - y0) / (y1 - y0) * (h - pad_t - pad_b)

    parts = [f'<svg viewBox="0 0 {w} {h}" width="100%" height="{h}" '
             f'role="img" aria-label="{html.escape(title)}">', head]
    # y grid + labels
    for tick in _ticks(y0, y1, 5):
        yy = sy(tick)
        parts.append(
            f'<line x1="{pad_l:.1f}" y1="{yy:.1f}" x2="{w - pad_r:.1f}" '
            f'y2="{yy:.1f}" stroke="{GRID}" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{pad_l - 6:.1f}" y="{yy + 3.5:.1f}" font-size="10" '
            f'text-anchor="end" fill="{MUTED}">{html.escape(fmt(tick, y_digits))}'
            f"</text>"
        )
    # zero line
    if y0 < 0 < y1:
        yy = sy(0.0)
        parts.append(
            f'<line x1="{pad_l:.1f}" y1="{yy:.1f}" x2="{w - pad_r:.1f}" '
            f'y2="{yy:.1f}" stroke="#b9b9b9" stroke-dasharray="3 3"/>'
        )
    # x ticks
    for tick in _ticks(x0, x1, 5):
        xx = sx(tick)
        parts.append(
            f'<text x="{xx:.1f}" y="{h - pad_b + 14:.1f}" font-size="10" '
            f'text-anchor="middle" fill="{MUTED}">{int(round(tick)) if tick >= 0 else tick}'
            f"</text>"
        )
    parts.append(
        f'<text x="{w - pad_r:.1f}" y="{h - 4}" font-size="10" '
        f'text-anchor="end" fill="{MUTED}">{html.escape(x_label)}</text>'
    )
    # series
    for i, s in enumerate(usable):
        color = s.get("color") or PALETTE[i % len(PALETTE)]
        pts = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in s["pts"])
        dash = ' stroke-dasharray="5 3"' if s.get("dashed") else ""
        parts.append(
            f'<polyline points="{pts}" fill="none" stroke="{color}" '
            f'stroke-width="{s.get("width", 1.8)}"{dash} '
            f'stroke-linejoin="round"/>'
        )
        lx, ly = s["pts"][-1]
        parts.append(
            f'<circle cx="{sx(lx):.1f}" cy="{sy(ly):.1f}" r="2.6" '
            f'fill="{color}"/>'
        )
        label = s.get("label", f"series {i + 1}")
        parts.append(
            f'<text x="{sx(lx) - 4:.1f}" y="{sy(ly) - 7:.1f}" font-size="10" '
            f'text-anchor="end" fill="{color}">{html.escape(fmt(ly, y_digits))}'
            f"</text>"
        )
    # legend
    lx = 8
    for i, s in enumerate(usable):
        color = s.get("color") or PALETTE[i % len(PALETTE)]
        label = s.get("label", f"series {i + 1}")
        last = s["pts"][-1][1]
        text = f"{label} {fmt(last, y_digits)}"
        parts.append(
            f'<rect x="{lx}" y="23" width="8" height="8" rx="1.5" '
            f'fill="{color}"/>'
        )
        parts.append(
            f'<text x="{lx + 12}" y="31" font-size="10.5" fill="{MUTED}">'
            f"{html.escape(text)}</text>"
        )
        lx += 14 + 6.0 * len(text)
    parts.append("</svg>")
    return "".join(parts)


def svg_hist(counts: list[int], lo: float, hi: float, *, title: str,
             w: int = 560, h: int = 170, x_digits: int = 3,
             bar_color: str = "#2f6f9f") -> str:
    """Histogram of an immediate variable (fixed bins from the registry)."""
    head = (
        f'<text x="8" y="16" font-size="12.5" font-weight="600" '
        f'fill="{INK}">{html.escape(title)}</text>'
    )
    total = sum(counts)
    if not total:
        return (
            f'<svg viewBox="0 0 {w} {h}" width="100%" height="{h}" '
            f'role="img" aria-label="{html.escape(title)}">{head}'
            f'<text x="8" y="{h // 2}" font-size="12" fill="{MUTED}">'
            f"no data yet</text></svg>"
        )
    pad_l, pad_r, pad_t, pad_b = 52, 14, 30, 30
    bins = len(counts)
    plot_w = w - pad_l - pad_r
    plot_h = h - pad_t - pad_b
    peak = max(counts)
    bw = plot_w / bins
    parts = [f'<svg viewBox="0 0 {w} {h}" width="100%" height="{h}" '
             f'role="img" aria-label="{html.escape(title)}">', head]
    for tick in (0.0, 0.25, 0.5, 0.75, 1.0):
        yy = pad_t + plot_h - tick * plot_h
        parts.append(
            f'<line x1="{pad_l}" y1="{yy:.1f}" x2="{w - pad_r}" y2="{yy:.1f}" '
            f'stroke="{GRID}"/>'
        )
        parts.append(
            f'<text x="{pad_l - 6}" y="{yy + 3.5:.1f}" font-size="10" '
            f'text-anchor="end" fill="{MUTED}">{int(round(tick * peak))}</text>'
        )
    for i, count in enumerate(counts):
        bh = (count / peak) * plot_h if peak else 0
        x = pad_l + i * bw
        parts.append(
            f'<rect x="{x + 0.5:.1f}" y="{pad_t + plot_h - bh:.1f}" '
            f'width="{max(1.0, bw - 1.0):.1f}" height="{bh:.1f}" '
            f'fill="{bar_color}" opacity="0.85"/>'
        )
    for i in range(0, bins + 1, max(1, bins // 5)):
        edge = lo + (hi - lo) * i / bins
        parts.append(
            f'<text x="{pad_l + i * bw:.1f}" y="{h - pad_b + 14}" '
            f'font-size="10" text-anchor="middle" fill="{MUTED}">'
            f"{html.escape(fmt(edge, x_digits))}</text>"
        )
    parts.append(
        f'<text x="{w - pad_r}" y="{h - 4}" font-size="10" text-anchor="end" '
        f'fill="{MUTED}">n={total}</text>'
    )
    parts.append("</svg>")
    return "".join(parts)


def svg_bars(labels: list[str], values: list[float], *, title: str,
             w: int = 560, h: int | None = None, value_digits: int = 4,
             color: str = "#4f8a3d", annotate: bool = True) -> str:
    """Horizontal bar chart (used for the per-strategy value table)."""
    if not labels:
        return svg_line([], title=title, h=140)
    row = 20
    h = h or (34 + row * len(labels))
    pad_l, pad_r, pad_t = 8, 76, 30
    plot_w = w - pad_l - pad_r
    lo = min(0.0, min(values))
    hi = max(0.0, max(values)) or 1.0
    head = (
        f'<text x="8" y="16" font-size="12.5" font-weight="600" '
        f'fill="{INK}">{html.escape(title)}</text>'
    )
    parts = [f'<svg viewBox="0 0 {w} {h}" width="100%" height="{h}" '
             f'role="img" aria-label="{html.escape(title)}">', head]

    def sx(v):
        return pad_l + (v - lo) / (hi - lo) * plot_w

    zero = sx(0.0)
    parts.append(
        f'<line x1="{zero:.1f}" y1="{pad_t - 6}" x2="{zero:.1f}" '
        f'y2="{h - 6}" stroke="#c9c9c9"/>'
    )
    for i, (label, value) in enumerate(zip(labels, values)):
        y = pad_t + i * row
        x0, x1 = (zero, sx(value)) if value >= 0 else (sx(value), zero)
        parts.append(
            f'<rect x="{x0:.1f}" y="{y:.1f}" width="{max(1.0, x1 - x0):.1f}" '
            f'height="{row - 7}" fill="{color}" opacity="0.85"/>'
        )
        parts.append(
            f'<text x="{pad_l}" y="{y + row - 11:.1f}" font-size="9.5" '
            f'fill="{INK}">{html.escape(label)}</text>'
        )
        if annotate:
            parts.append(
                f'<text x="{w - pad_r + 4}" y="{y + row - 11:.1f}" '
                f'font-size="9.5" fill="{MUTED}">'
                f"{html.escape(fmt(value, value_digits))}</text>"
            )
    parts.append("</svg>")
    return "".join(parts)


def svg_spark(values: list[float], *, w: int = 150, h: int = 34,
              color: str = "#2f6f9f") -> str:
    if len(values) < 2:
        return f'<svg viewBox="0 0 {w} {h}" width="{w}" height="{h}"></svg>'
    lo, hi = min(values), max(values)
    if hi == lo:
        hi = lo + 1e-9
    pts = " ".join(
        f"{i / (len(values) - 1) * (w - 4) + 2:.1f},"
        f"{h - 3 - (v - lo) / (hi - lo) * (h - 6):.1f}"
        for i, v in enumerate(values)
    )
    return (
        f'<svg viewBox="0 0 {w} {h}" width="{w}" height="{h}" role="img">'
        f'<polyline points="{pts}" fill="none" stroke="{color}" '
        f'stroke-width="1.4"/></svg>'
    )


# ----------------------------------------------------------------------
# dashboard sections
# ----------------------------------------------------------------------
KPI_SPECS: tuple[tuple[str, str, str, int], ...] = (
    ("Episode reward", "mean_episode_reward", "iteration", 4),
    ("Persuasiveness (judge)", "train_judge.mean", "iteration", 4),
    ("Judge saturated", "train_judge.saturated_frac", "iteration", 4),
    ("Goal alignment a(m)", "train_goal_align", "iteration", 4),
    ("Final stance", "mean_final_stance", "iteration", 4),
    ("Adopters", "mean_adopted", "iteration", 2),
    ("Policy entropy", "train_entropy", "iteration", 4),
    ("KL(old‖new)", "train_kl", "iteration", 5),
    ("Entropy coef", "train_entropy_coef", "iteration", 4),
    ("Strategy entropy (cum)", "train_strategy_entropy", "iteration", 4),
)

TRACE_KPI_SPECS: tuple[tuple[str, str, str, int], ...] = (
    ("Message persuasiveness", "steps.persuasiveness", "steps", 4),
    ("Message goal_align", "steps.goal_align", "steps", 4),
    ("Per-persuader reward", "steps.reward", "steps", 4),
    ("Influence probability", "events.influence_probability", "events", 4),
    ("Stance delta per hop", "events.target_delta_stance", "events", 4),
    ("Adopters per period", "periods.n_adopted", "periods", 2),
    ("Mask open fraction", "steps.mask_open_frac", "steps", 3),
)


def _spark_from(series: list[tuple[float, float]], n: int = 60) -> str:
    if not series:
        return ""
    return svg_spark([v for _, v in series[-n:]])


def render_kpis(rs: RunState) -> str:
    cards = []
    for label, name, _stream, digits in KPI_SPECS:
        series = rs.series(name)
        stats = rs.history_stats(name)
        if not series and stats["last"] is None:
            continue
        cards.append(
            f'<div class="card"><div class="k">{html.escape(label)}</div>'
            f'<div class="v" title="{html.escape(fmt(stats["last"], 12))}">'
            f'{html.escape(fmt(stats["last"], digits))}</div>'
            f'<div class="sub">min {html.escape(fmt(stats["min"], digits))} · '
            f'mean {html.escape(fmt(stats["mean"], digits))} · '
            f'max {html.escape(fmt(stats["max"], digits))}</div>'
            f'<div class="code">{html.escape(name)}</div>'
            f"<div>{_spark_from(series)}</div></div>"
        )
    for label, key, stream, digits in TRACE_KPI_SPECS:
        tail = rs.tails.get(stream)
        if tail is None or not tail.exists or not tail.records:
            continue
        name = key.split(".", 1)[1]
        stats = tail.stats(name)
        if stats["n"] == 0:
            continue
        cards.append(
            f'<div class="card"><div class="k">{html.escape(label)}</div>'
            f'<div class="v" title="{html.escape(fmt(stats["last"], 12))}">'
            f'{html.escape(fmt(stats["last"], digits))}</div>'
            f'<div class="sub">min {html.escape(fmt(stats["min"], digits))} · '
            f'mean {html.escape(fmt(stats["mean"], digits))} · '
            f'max {html.escape(fmt(stats["max"], digits))}</div>'
            f'<div class="code">{html.escape(key)} (n={stats["n"]:,})</div>'
            f'<div>{_spark_from(tail.iter_series(name))}</div></div>'
        )
    if not cards:
        return '<p class="warn">No metrics yet — is the run writing to this directory?</p>'
    return '<div class="cards">' + "".join(cards) + "</div>"


def render_status(rs: RunState) -> str:
    st = rs.status()
    rows = [
        ("run directory", st["run_dir"]),
        ("mechanism", st["mechanism"]),
        ("LLM model", st["model"]),
        ("iteration", f'{st["iteration"]} / {st["iterations_total"]}'),
        ("elapsed", fmt_dur(st["elapsed_s"])),
        ("rate", None if st["iters_per_min"] is None
         else f'{st["iters_per_min"]:.2f} iters/min'),
        ("ETA", fmt_dur(st["eta_s"])),
        ("log age", fmt_age(st["updated_age_s"])),
        ("trace streams", ", ".join(st["trace_streams"]) or
         "none — this run was started without --trace"),
        ("population", f'{st["agent_num"]} agents, '
                       f'{st["n_persuaders"]} persuaders, {st["topology"]}'),
        ("episodes/iteration", st["episodes_per_iteration"]),
        ("workers", st["workers"]),
        ("periods", st["n_periods"]),
        ("target selection", f'{st["target_selection"]} / obs {st["obs_mode"]}'),
        ("reward model", st["judge"]),
        ("seed", st["seed"]),
    ]
    live = ""
    if st["updated_age_s"] is not None:
        live = (
            f'<span class="dot ok"></span> live'
            if st["updated_age_s"] < 900
            else f'<span class="dot stale"></span> stale'
        )
    body = "".join(
        f'<div class="srow"><span class="sk">{html.escape(str(k))}</span>'
        f'<span class="sv">{html.escape(fmt(v))}</span></div>'
        for k, v in rows if v is not None
    )
    return (
        f'<div class="card wide"><div class="k">Run status {live}</div>'
        f'<div class="status">{body}</div></div>'
    )


def render_charts(rs: RunState) -> str:
    charts = []

    def line(title, specs, y_digits=4, x_label="iteration"):
        series = []
        for i, (label, name, dashed) in enumerate(specs):
            pts = rs.series(name)
            if pts:
                series.append({
                    "label": label, "pts": pts, "dashed": dashed,
                    "color": PALETTE[i % len(PALETTE)],
                })
        return svg_line(series, title=title, y_digits=y_digits,
                        x_label=x_label)

    charts.append(line("Learning curve — mean episode reward", [
        ("reward", "mean_episode_reward", False),
    ]))
    charts.append(line("Population persuasion", [
        ("mean final stance", "mean_final_stance", False),
        ("mean adopted", "mean_adopted", True),
    ]))
    charts.append(line("Judge score of delivered messages", [
        ("mean", "train_judge.mean", False),
        ("min", "train_judge.min", True),
        ("max", "train_judge.max", True),
    ]))
    charts.append(line("Judge saturation (>= 0.95)", [
        ("train", "train_judge.saturated_frac", False),
        ("eval", "eval_judge.saturated_frac", True),
    ]))
    charts.append(line("Judge lookup path", [
        ("identity index", "train_judge_src.id", False),
        ("content fallback", "train_judge_src.text", True),
        ("none (idle)", "train_judge_src.none", True),
    ]))
    charts.append(line("Goal alignment a(m)", [
        ("iteration mean", "train_goal_align", False),
    ]))
    charts.append(line("Policy entropy", [
        ("train", "train_entropy", False),
        ("eval action H", "eval_action_entropy", True),
    ]))
    charts.append(line("Exploration breadth", [
        ("strategy entropy (cum)", "train_strategy_entropy", False),
        ("eval strategy H", "eval_strategy_entropy", True),
    ]))
    charts.append(line("Losses", [
        ("policy", "train_pi_loss", False),
        ("value", "train_v_loss", True),
    ], y_digits=5))
    charts.append(line("Update size", [
        ("KL(old||new)", "train_kl", False),
        ("entropy coef", "train_entropy_coef", True),
    ], y_digits=5))
    charts.append(line("Paired evaluation (shared seeds)", [
        ("greedy", "paired_eval.greedy", False),
        ("random", "paired_eval.random", True),
        ("heuristic", "paired_eval.heuristic", True),
    ]))
    charts.append(line("Paired delta vs random", [
        ("greedy - random", "paired_eval.vs_random.mean", False),
        ("greedy - heuristic", "paired_eval.vs_heuristic.mean", True),
    ]))

    # immediate variables, charted against the training curve
    trace_specs = (
        ("IM: message persuasiveness (mean/iter)", "steps", "persuasiveness",
         ("persuasiveness", False)),
        ("IM: goal alignment (mean/iter)", "steps", "goal_align",
         ("goal_align", False)),
        ("IM: per-persuader reward (mean/iter)", "steps", "reward",
         ("reward", False)),
        ("IM: influence probability (mean/iter)", "events",
         "influence_probability", ("influence_probability", False)),
        ("IM: stance delta per hop (mean/iter)", "events",
         "target_delta_stance", ("target_delta_stance", False)),
    )
    for title, stream, name, (label, dashed) in trace_specs:
        tail = rs.tails.get(stream)
        if tail is None or not tail.records:
            continue
        pts = tail.iter_series(name)
        if pts:
            charts.append(svg_line(
                [{"label": label, "pts": pts,
                  "color": PALETTE[0], "dashed": dashed}],
                title=title, y_digits=4,
            ))
    # within-episode trajectory (averaged over every traced episode)
    for title, name, y_digits in (
        ("Within-episode: mean stance by period", "mean_stance", 4),
        ("Within-episode: adopters by period", "n_adopted", 2),
        ("Within-episode: persuader events by period", "n_persuader_events", 2),
        ("Within-episode: accepts by period", "n_accept", 2),
    ):
        tail = rs.tails.get("periods")
        if tail is None or not tail.records:
            continue
        pts = tail.period_series(name)
        if pts:
            charts.append(svg_line(
                [{"label": name, "pts": pts}], title=title,
                y_digits=y_digits, x_label="period",
            ))

    # histograms of immediate variables
    for stream in ("steps", "events", "periods", "episodes"):
        tail = rs.tails.get(stream)
        if tail is None or not tail.records:
            continue
        for field in HISTOGRAMS.get(stream, ()):
            counts = tail.hists.get(field.name)
            if not counts or not sum(counts):
                continue
            lo, hi, _bins = field.hist
            charts.append(svg_hist(
                counts, lo, hi,
                title=f"{stream}.{field.name} distribution "
                      f"({field.unit or 'value'})",
            ))
    return '<div class="grid">' + "".join(
        f'<div class="chart">{c}</div>' for c in charts
    ) + "</div>"


def render_strategy(rs: RunState) -> str:
    rows = rs.strategy_value.get("rows") or []
    out = []
    if rows:
        top = sorted(rows, key=lambda r: -(r.get("count") or 0))[:15]
        labels = [
            f'{i + 1}. {r.get("strategy", "")[:46]} (n={r.get("count")})'
            for i, r in enumerate(top)
        ]
        values = [float(r.get("mean_reward") or 0.0) for r in top]
        out.append(svg_bars(
            labels, values,
            title="Mean reward per strategy arm (top 15 by usage; whole run)",
        ))
        body = "".join(
            "<tr>"
            f'<td>{r.get("strategy_index")}</td>'
            f'<td class="mono">{html.escape(str(r.get("strategy", "")))}</td>'
            f'<td class="num">{r.get("count")}</td>'
            f'<td class="num">{html.escape(fmt(r.get("mean_reward"), 6))}</td>'
            f'<td class="num">{html.escape(fmt(r.get("total_reward"), 6))}</td>'
            "</tr>"
            for r in sorted(rows, key=lambda r: -(r.get("mean_reward") or 0))
        )
        out.append(
            '<div class="tablewrap"><table><thead><tr>'
            "<th>idx</th><th>strategy</th><th>count</th>"
            "<th>mean reward</th><th>total reward</th></tr></thead>"
            f"<tbody>{body}</tbody></table></div>"
        )
        out.append(
            f'<p class="note">Cumulative strategy entropy '
            f'{html.escape(fmt(rs.strategy_value.get("strategy_entropy"), 6))} '
            f'nats over {thousands(rs.strategy_value.get("n_decisions"))} '
            f'decisions. Sorted by mean reward; the policy is learning this '
            f'table implicitly.</p>'
        )
    else:
        out.append('<p class="note">No strategy value table yet '
                   '(written every eval interval).</p>')
    return "".join(out)


def thousands(n) -> str:
    try:
        return f"{int(n):,}"
    except (TypeError, ValueError):
        return fmt(n)


def _immediate_tables(rs: RunState) -> str:
    out = []
    steps = rs.tails.get("steps")
    if steps is not None and steps.records:
        recent = list(steps.recent)[-25:][::-1]
        body = "".join(
            "<tr>"
            f'<td class="num">{r.get("iteration")}</td>'
            f'<td class="num">{r.get("episode")}</td>'
            f'<td class="num">{r.get("period")}</td>'
            f'<td class="num">{r.get("persuader")}</td>'
            f'<td class="num">{r.get("strategy_index")}</td>'
            f'<td class="num">{r.get("target") if r.get("target") is not None else "—"}</td>'
            f'<td class="num">{html.escape(fmt(r.get("persuasiveness"), 5))}</td>'
            f'<td class="num">{html.escape(fmt(r.get("goal_align"), 5))}</td>'
            f'<td class="num">{html.escape(fmt(r.get("reward"), 5))}</td>'
            f'<td>{html.escape(str(r.get("judge_src")))}</td>'
            f'<td class="msg">{html.escape(str(r.get("message") or "")[:110])}</td>'
            "</tr>"
            for r in recent
        )
        out.append(
            '<h3>Immediate variables — last persuader decisions '
            f'<span class="hint">(trace/steps.jsonl, newest first; '
            f'{thousands(steps.records)} records so far)</span></h3>'
            '<div class="tablewrap"><table><thead><tr>'
            "<th>iter</th><th>ep</th><th>period</th><th>persuader</th>"
            "<th>strat idx</th><th>target</th><th>persuasiveness</th>"
            "<th>goal_align</th><th>reward</th><th>judge src</th>"
            "<th>message</th></tr></thead>"
            f"<tbody>{body}</tbody></table></div>"
        )
    periods = rs.tails.get("periods")
    if periods is not None and periods.records:
        recent = list(periods.recent)[-18:][::-1]
        body = "".join(
            "<tr>"
            f'<td class="num">{r.get("iteration")}</td>'
            f'<td class="num">{r.get("episode")}</td>'
            f'<td class="num">{r.get("period")}</td>'
            f'<td class="num">{html.escape(fmt(r.get("mean_stance"), 5))}</td>'
            f'<td class="num">{html.escape(fmt(r.get("n_adopted"), 8))}</td>'
            f'<td class="num">{html.escape(fmt(r.get("n_persuader_events"), 8))}</td>'
            f'<td class="num">{html.escape(fmt(r.get("n_accept"), 8))}</td>'
            f'<td class="num">{html.escape(fmt(r.get("n_accept_all"), 8))}</td>'
            f'<td class="num">{html.escape(fmt(r.get("n_events"), 8))}</td>'
            f'<td class="num">{html.escape(fmt(r.get("mask_open_frac"), 4))}</td>'
            f'<td class="num">{html.escape(fmt(r.get("period_reward_sum"), 5))}</td>'
            "</tr>"
            for r in recent
        )
        out.append(
            '<h3>Population state — last periods '
            '<span class="hint">(trace/periods.jsonl; n_accept counts '
            'persuader-sourced accepts, n_accept_all every source)</span></h3>'
            '<div class="tablewrap"><table><thead><tr>'
            "<th>iter</th><th>ep</th><th>period</th><th>mean stance</th>"
            "<th>adopted</th><th>persuader events</th><th>accepts (P)</th>"
            "<th>accepts (all)</th><th>events</th><th>mask open</th>"
            "<th>period reward Σ</th></tr></thead>"
            f"<tbody>{body}</tbody></table></div>"
        )
    events = rs.tails.get("events")
    if events is not None and events.records:
        recent = list(events.recent)[-18:][::-1]
        body = "".join(
            "<tr>"
            f'<td class="num">{r.get("iteration")}</td>'
            f'<td class="num">{r.get("episode")}</td>'
            f'<td class="num">{r.get("period")}</td>'
            f'<td class="num">{r.get("source_agent")}</td>'
            f'<td>{"yes" if r.get("source_is_persuader") else "no"}</td>'
            f'<td class="num">{r.get("target_agent")}</td>'
            f'<td class="num">{html.escape(fmt(r.get("target_stance_before"), 4))}</td>'
            f'<td class="num">{html.escape(fmt(r.get("target_stance_after"), 4))}</td>'
            f'<td class="num">{html.escape(fmt(r.get("target_delta_stance"), 5))}</td>'
            f'<td class="num">{html.escape(fmt(r.get("influence_probability"), 4))}</td>'
            f'<td>{html.escape(str(r.get("behavior")))}</td>'
            f'<td class="num">{html.escape(fmt(r.get("adopted"), 8))}</td>'
            "</tr>"
            for r in recent
        )
        out.append(
            '<h3>Diffusion events — last hops '
            f'<span class="hint">(trace/events.jsonl, '
            f'{thousands(events.records)} records so far)</span></h3>'
            '<div class="tablewrap"><table><thead><tr>'
            "<th>iter</th><th>ep</th><th>period</th><th>source</th>"
            "<th>persuader?</th><th>target</th><th>stance before</th>"
            "<th>stance after</th><th>Δstance</th><th>P(influence)</th>"
            "<th>behavior</th><th>adopted</th></tr></thead>"
            f"<tbody>{body}</tbody></table></div>"
        )
    if not out:
        out.append(
            '<p class="warn">No immediate-variable trace for this run. '
            "Training was started without <code>--trace</code>, so "
            "per-decision/per-event values were not recorded. Restart or "
            "resume with <code>--trace</code> to collect them.</p>"
        )
    return "".join(out)


def variable_rows(rs: RunState) -> list[dict]:
    """Every registered variable with exact statistics."""
    rows = []
    for stream, fields in STREAMS.items():
        for field in fields:
            row = {
                "stream": stream,
                "name": field.name,
                "kind": field.kind,
                "unit": field.unit,
                "desc": field.desc,
                "n": None, "last": None, "min": None, "max": None,
                "mean": None,
            }
            if stream in ("iteration", "eval", "paired_eval"):
                row.update(rs.history_stats(field.name))
                if field.kind in ("cat", "str", "list"):
                    value = rs.last_value(field.name)
                    row.update(n=None, last=value, min=None, max=None,
                               mean=None)
            elif stream in TRACE_STREAMS:
                tail = rs.tails.get(stream)
                if tail is None or not tail.records:
                    row["note"] = "not traced in this run"
                else:
                    name = field.name
                    if field.kind in ("num", "flag"):
                        row.update(tail.stats(name))
                    elif field.kind == "cat":
                        counter = tail.cats.get(name)
                        if counter:
                            row["last"] = counter.most_common(1)[0][0]
                            row["note"] = f"{len(counter)} distinct"
                    elif tail.recent:
                        row["last"] = tail.recent[-1].get(name)
            elif stream == "strategy":
                rows_sv = rs.strategy_value.get("rows") or []
                if field.name in ("strategy_index", "strategy", "count",
                                  "mean_reward", "total_reward"):
                    row["note"] = "per-arm — see the table"
                    row["n"] = len(rows_sv)
                elif field.name in ("strategy_entropy", "n_decisions"):
                    row["last"] = rs.strategy_value.get(field.name)
            elif stream == "status":
                row["last"] = rs.status().get(field.name)
            rows.append(row)
    return rows


def render_registry(rs: RunState) -> str:
    groups = {}
    for row in variable_rows(rs):
        groups.setdefault(row["stream"], []).append(row)
    out = [
        "<h3>All variables and their descriptions "
        '<span class="hint">(single source of truth: '
        "code/metrics_registry.py — also exported to METRICS.md)</span></h3>"
    ]
    for stream, rows in groups.items():
        body = "".join(
            "<tr>"
            f'<td class="mono">{html.escape(r["name"])}</td>'
            f'<td>{html.escape(r["kind"])}</td>'
            f'<td>{html.escape(r["unit"] or "—")}</td>'
            f'<td class="num">{html.escape(fmt(r["last"], 8))}</td>'
            f'<td class="num">{html.escape(fmt(r["min"], 8))}</td>'
            f'<td class="num">{html.escape(fmt(r["max"], 8))}</td>'
            f'<td class="num">{html.escape(fmt(r["mean"], 8))}</td>'
            f'<td class="num">{"" if r["n"] is None else thousands(r["n"])}</td>'
            f'<td class="desc">{html.escape(r["desc"])}'
            + (f' <span class="hint">{html.escape(r["note"])}</span>'
               if r.get("note") else "")
            + "</td></tr>"
            for r in rows
        )
        out.append(
            f'<h4>{html.escape(stream)}</h4>'
            '<div class="tablewrap"><table class="reg"><thead><tr>'
            "<th>variable</th><th>kind</th><th>unit</th><th>current</th>"
            "<th>min</th><th>max</th><th>mean</th><th>n</th>"
            "<th>description</th></tr></thead>"
            f"<tbody>{body}</tbody></table></div>"
        )
    return "".join(out)


def render_body(rs: RunState) -> str:
    return (
        render_status(rs)
        + "<h3>Live values</h3>"
        + render_kpis(rs)
        + "<h3>Charts</h3>"
        + render_charts(rs)
        + "<h3>Strategy value table</h3>"
        + render_strategy(rs)
        + _immediate_tables(rs)
        + render_registry(rs)
    )


CSS = """
:root { color-scheme: light; }
body { font: 13px/1.5 -apple-system, "Segoe UI", Roboto, Helvetica, Arial,
       sans-serif; color: #1a1a1a; margin: 0; background: #fafafa; }
main { max-width: 1720px; margin: 0 auto; padding: 18px 22px 80px; }
h1 { font-size: 19px; margin: 0 0 2px; }
h3 { font-size: 14.5px; margin: 26px 0 8px; border-bottom: 1px solid #e3e3e3;
     padding-bottom: 4px; }
h4 { font-size: 12.5px; margin: 16px 0 6px; color: #444; text-transform:
     uppercase; letter-spacing: .04em; }
.hint { color: #8a8a8a; font-weight: 400; font-size: 11.5px; }
.cards { display: grid; grid-template-columns: repeat(auto-fill,
         minmax(196px, 1fr)); gap: 10px; }
.card { background: #fff; border: 1px solid #e3e3e3; border-radius: 7px;
        padding: 9px 11px; }
.card.wide { grid-column: 1 / -1; }
.card .k { font-size: 11px; color: #6b6b6b; text-transform: uppercase;
           letter-spacing: .03em; }
.card .v { font-size: 21px; font-weight: 600; font-variant-numeric:
           tabular-nums; }
.card .sub { font-size: 10.5px; color: #6b6b6b; font-variant-numeric:
             tabular-nums; }
.card .code { font: 10px ui-monospace, Menlo, Consolas, monospace;
              color: #9a9a9a; margin-top: 2px; }
.status { display: grid; grid-template-columns: repeat(auto-fill,
          minmax(290px, 1fr)); gap: 2px 18px; }
.srow { display: flex; justify-content: space-between; gap: 12px;
        border-bottom: 1px dotted #eee; padding: 1.5px 0; }
.sk { color: #6b6b6b; }
.sv { font-variant-numeric: tabular-nums; }
.dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%;
       margin-left: 6px; }
.dot.ok { background: #4f8a3d; } .dot.stale { background: #c0504d; }
.grid { display: grid; grid-template-columns: repeat(auto-fill,
         minmax(430px, 1fr)); gap: 12px; }
.chart { background: #fff; border: 1px solid #e3e3e3; border-radius: 7px;
         padding: 6px 8px 2px; }
.tablewrap { overflow-x: auto; background: #fff; border: 1px solid #e3e3e3;
             border-radius: 7px; }
table { border-collapse: collapse; width: 100%; font-size: 11.5px; }
th, td { padding: 3.5px 7px; border-bottom: 1px solid #f0f0f0;
         text-align: left; vertical-align: top; }
th { background: #f6f6f6; position: sticky; top: 0; font-weight: 600;
     font-size: 11px; }
td.num { text-align: right; font-variant-numeric: tabular-nums;
         white-space: nowrap; }
td.mono { font-family: ui-monospace, Menlo, Consolas, monospace;
          font-size: 11px; }
td.desc { max-width: 620px; color: #333; }
td.msg { max-width: 330px; color: #555; }
tr:hover td { background: #fbfbf6; }
.note { color: #6b6b6b; font-size: 11.5px; }
.warn { background: #fff7e6; border: 1px solid #f0d9a8; padding: 8px 11px;
        border-radius: 7px; }
code { background: #f2f2f2; padding: 1px 4px; border-radius: 3px;
       font-size: 11.5px; }
#bar { position: sticky; top: 0; z-index: 5; background: #fffffff2;
       backdrop-filter: blur(4px); border-bottom: 1px solid #e3e3e3;
       padding: 8px 22px; margin: 0 -22px 12px; display: flex;
       gap: 14px; align-items: baseline; }
#bar .sp { flex: 1; }
#bar .t { font-variant-numeric: tabular-nums; color: #6b6b6b; font-size: 11.5px; }
"""

JS = """
let paused = false, seen = null;
async function tick() {
  if (paused) return;
  try {
    const r = await fetch('partial' + location.search, {cache: 'no-store'});
    if (r.ok) { document.getElementById('live').innerHTML = await r.text(); }
  } catch (e) {}
}
setInterval(tick, REFRESH_MS);
document.addEventListener('DOMContentLoaded', () => {
  const b = document.getElementById('pause');
  if (b) b.onclick = () => { paused = !paused;
    b.textContent = paused ? 'resume' : 'pause'; };
  tick();
});
"""


def render_page(rs: RunState, refresh_s: int = 15) -> str:
    st = rs.status()
    title = f'{Path(st["run_dir"]).name or st["run_dir"]} — {st["mechanism"] or "run"} monitor'
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<style>{CSS}</style></head>
<body><main>
<div id="bar">
  <div><h1>{html.escape(title)}</h1>
  <div class="t">exact values, recomputed from every record — no sampling
  · refresh {refresh_s}s</div></div>
  <div class="sp"></div>
  <div class="t" id="clock"></div>
  <button id="pause" type="button">pause</button>
</div>
<div id="live">{render_body(rs)}</div>
</main>
<script>const REFRESH_MS = {refresh_s * 1000};{JS}
document.getElementById('clock').textContent = new Date().toLocaleTimeString();
setInterval(() => document.getElementById('clock').textContent =
  new Date().toLocaleTimeString(), 1000);
</script>
</body></html>"""


def state_payload(rs: RunState) -> dict:
    status = rs.status()
    series = {
        f.name: rs.series(f.name)
        for stream in ("iteration", "eval", "paired_eval")
        for f in chart_fields(stream)
    }
    traces = {}
    for stream, tail in rs.tails.items():
        if not tail.exists:
            continue
        traces[stream] = {
            "path": str(tail.path),
            "records": tail.records,
            "bad_lines": tail.bad_lines,
            "numeric": {n: s.as_dict() for n, s in tail.numeric.items()},
            "categorical": {
                n: dict(c.most_common(20)) for n, c in tail.cats.items()
            },
            "histograms": {
                f.name: {
                    "lo": f.hist[0], "hi": f.hist[1], "bins": f.hist[2],
                    "counts": tail.hists.get(f.name, []),
                }
                for f in HISTOGRAMS.get(stream, ())
            },
            "per_iteration_mean": {
                name: tail.iter_series(name)
                for name in ITER_AGGREGATE.get(stream, ())
            },
            "per_period_mean": {
                name: tail.period_series(name)
                for name in ("mean_stance", "n_adopted", "n_accept",
                             "n_persuader_events", "n_events")
            } if stream == "periods" else {},
            "per_strategy": tail.by_strategy if stream == "steps" else {},
            "recent": list(tail.recent),
        }
    return {
        "generated_at": time.time(),
        "status": status,
        "config": rs.config,
        "trace_meta": rs.meta,
        "iterations": rs.history,
        "results": rs.results,
        "strategy_value": rs.strategy_value,
        "series": series,
        "traces": traces,
        "variables": variable_rows(rs),
    }


def render_text(rs: RunState, width: int = 118) -> str:
    st = rs.status()
    rate = "—" if st["iters_per_min"] is None else f'{st["iters_per_min"]:.2f} iters/min'
    out = []
    add = out.append
    add("=" * width)
    add(f'RUN  {st["run_dir"]}')
    add(f'  mechanism={st["mechanism"]}  model={st["model"]}')
    add(f'  iteration={st["iteration"]}/{st["iterations_total"]}  '
        f'elapsed={fmt_dur(st["elapsed_s"])}  rate={rate}  '
        f'ETA={fmt_dur(st["eta_s"])}  log age={fmt_age(st["updated_age_s"])}')
    add(f'  trace streams: '
        f'{", ".join(st["trace_streams"]) or "none (run without --trace)"}')
    add("")
    add("LIVE VALUES (exact)")
    for label, name, stream, digits in KPI_SPECS:
        if stream in ("iteration", "eval", "paired_eval"):
            stats = rs.history_stats(name)
        else:
            stats = _trace_stats(rs, stream, name)
        if stats["n"] in (0, None):
            continue
        add(f'  {label:26s} {fmt(stats["last"], digits):>14s}   '
            f'min {fmt(stats["min"], digits):>12s}  mean {fmt(stats["mean"], digits):>12s}  '
            f'max {fmt(stats["max"], digits):>12s}  n={thousands(stats["n"])}   [{name}]')
    for label, key, stream, digits in TRACE_KPI_SPECS:
        tail = rs.tails.get(stream)
        if tail is None or not tail.records:
            continue
        name = key.split(".", 1)[1]
        stats = tail.stats(name)
        if not stats["n"]:
            continue
        add(f'  {label:26s} {fmt(stats["last"], digits):>14s}   '
            f'min {fmt(stats["min"], digits):>12s}  mean {fmt(stats["mean"], digits):>12s}  '
            f'max {fmt(stats["max"], digits):>12s}  n={thousands(stats["n"])}   [{key}]')
    add("")
    add("PERIOD MEANS BY PERIOD (within-episode trajectory)")
    tail = rs.tails.get("periods")
    if tail is not None and tail.records:
        for name in ("mean_stance", "n_adopted", "n_persuader_events", "n_accept"):
            pts = tail.period_series(name)
            add(f'  {name:22s} ' + " ".join(f"{v:.3f}" for _, v in pts))
    else:
        add("  (not traced)")
    add("")
    add("LAST PERSUADER DECISIONS (immediate variables)")
    steps = rs.tails.get("steps")
    if steps is not None and steps.records:
        add("  iter  ep  per  pers  strat  target  persuasion  goal_align "
            "    reward  judge   message")
        for r in list(steps.recent)[-12:]:
            add(
                f'  {str(r.get("iteration")):>4s} {str(r.get("episode")):>3s} '
                f'{str(r.get("period")):>4s} {str(r.get("persuader")):>5s} '
                f'{str(r.get("strategy_index")):>6s} '
                f'{str(r.get("target")):>7s} '
                f'{fmt(r.get("persuasiveness"), 5):>11s} '
                f'{fmt(r.get("goal_align"), 5):>11s} '
                f'{fmt(r.get("reward"), 5):>10s} '
                f'{str(r.get("judge_src")):>6s}   '
                f'{str(r.get("message") or "")[:40]}'
            )
    else:
        add("  (not traced)")
    add("")
    add("VARIABLES  (stream · variable · unit · current · min · max · mean · n · description)")
    for row in variable_rows(rs):
        add(
            f'  {row["stream"]:11s} {row["name"]:24s} '
            f'{(row["unit"] or "-")[:18]:18s} '
            f'{fmt(row["last"], 8):>14s} {fmt(row["min"], 6):>10s} '
            f'{fmt(row["max"], 6):>10s} {fmt(row["mean"], 6):>10s} '
            f'{"" if row["n"] is None else thousands(row["n"]):>9s}  '
            f'{row["desc"][:70]}'
        )
    add("=" * width)
    return "\n".join(out)


# ----------------------------------------------------------------------
# server
# ----------------------------------------------------------------------
class Monitor:
    """Serialises polling (several browser tabs must not multiply work)."""

    def __init__(self, run_dir: str | Path, min_interval: float = 2.0,
                 keep_recent: int = 40):
        self.state = RunState(run_dir, keep_recent=keep_recent)
        self.state.poll()
        self.min_interval = min_interval
        self._lock = threading.Lock()
        self._last = 0.0

    def refresh(self, force: bool = False) -> RunState:
        with self._lock:
            now = time.time()
            if force or now - self._last >= self.min_interval:
                self.state.poll()
                self._last = now
        return self.state


def serve(monitor: Monitor, host: str, port: int, refresh_s: int) -> None:
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def _send(self, body: str, ctype: str = "text/html; charset=utf-8",
                  code: int = 200) -> None:
            payload = body.encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self):  # noqa: N802
            path = self.path.split("?", 1)[0]
            rs = monitor.refresh()
            if path in ("/", "/index.html"):
                self._send(render_page(rs, refresh_s))
            elif path == "/partial":
                self._send(render_body(rs))
            elif path == "/api/state":
                self._send(json.dumps(state_payload(rs), default=str),
                           "application/json; charset=utf-8")
            elif path == "/metrics.txt":
                self._send(render_text(rs), "text/plain; charset=utf-8")
            elif path == "/healthz":
                self._send("ok\n", "text/plain; charset=utf-8")
            else:
                self._send("not found\n", "text/plain; charset=utf-8", 404)

        def log_message(self, fmt, *args):  # quiet
            return

    httpd = ThreadingHTTPServer((host, port), Handler)
    st = monitor.state.status()
    print(f"[+] monitoring {st['run_dir']}")
    print(f"[+]     mechanism={st['mechanism']} "
          f"iteration={st['iteration']}/{st['iterations_total']} "
          f"trace={','.join(st['trace_streams']) or 'none'}")
    print(f"[+] dashboard  http://{host}:{port}/")
    print(f"[+] exact JSON http://{host}:{port}/api/state")
    print(f"[+] plain text http://{host}:{port}/metrics.txt")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[+] stopped")
    finally:
        httpd.server_close()


def find_newest_run(root: str | Path = "output") -> Path | None:
    """Newest ``training_log.json`` under `root` (by mtime)."""
    candidates = list(Path(root).glob("**/training_log.json"))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime).parent


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("run_dir", nargs="?", default=None,
                    help="run directory (default: newest under output/)")
    ap.add_argument("--port", type=int, default=3010)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--refresh", type=int, default=15,
                    help="browser refresh interval in seconds")
    ap.add_argument("--text", action="store_true",
                    help="print the terminal report and exit")
    ap.add_argument("--json", action="store_true",
                    help="print /api/state payload and exit")
    ap.add_argument("--snapshot", metavar="PATH",
                    help="write a static HTML snapshot and exit")
    ap.add_argument("--keep-recent", type=int, default=40,
                    help="raw records kept per stream for the tables")
    args = ap.parse_args()

    run_dir = Path(args.run_dir) if args.run_dir else find_newest_run()
    if run_dir is None or not run_dir.is_dir():
        print("error: no run directory found (pass one explicitly)")
        return 2
    monitor = Monitor(run_dir, keep_recent=args.keep_recent)
    monitor.refresh(force=True)

    if args.text:
        print(render_text(monitor.state))
        return 0
    if args.json:
        print(json.dumps(state_payload(monitor.state), indent=2, default=str))
        return 0
    if args.snapshot:
        Path(args.snapshot).write_text(render_page(monitor.state, args.refresh))
        print(f"[+] wrote {args.snapshot}")
        return 0
    serve(monitor, args.host, args.port, args.refresh)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
