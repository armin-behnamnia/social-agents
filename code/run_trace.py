"""Per-run instrumentation: append-only JSONL traces of what actually happened.

Why
---
`training_log.json` carries per-iteration *aggregates* (mean reward, mean
judge score, ...). That is enough to see a learning curve, but the
immediate variables — the persuasiveness of *this* message to *this*
target, the accept/reject outcome, each diffusion event, the stance it
moved — exist only inside the env's `info` dict for the duration of one
step and were thrown away. This module persists them.

Streams (all newline-delimited JSON, one directory per run):

- ``steps.jsonl``    one record per persuader decision (immediate vars)
- ``periods.jsonl``  one record per diffusion period per episode
- ``events.jsonl``   one record per diffusion event (hop)
- ``episodes.jsonl`` one record per rollout episode (summary)

plus ``meta.json`` with provenance (config, mechanism, start time,
version) so a trace can be interpreted months later.

Design notes
------------
- **Append-only and crash-safe**: records are written when an episode
  finishes, so an interrupted run keeps every completed episode. Resume
  appends to the same files.
- **Thread-safe**: rollouts run in a thread pool (160 workers in the
  campaign), so writes are serialised behind one lock. Batches (one per
  episode) keep contention negligible.
- **No dynamics**: tracing never mutates env state or rewards; it is a
  pure observer, so a traced run is comparable to an untraced one.
- Reading is `code/live_monitor.py`'s job (incremental tail + exact
  running aggregates); this module only writes.

Schema version: 1.
"""

from __future__ import annotations

import json
import threading
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

SCHEMA_VERSION = 1
STREAMS = ("episodes", "steps", "periods", "events")


class TraceWriter:
    """Append JSONL records to ``<directory>/<stream>.jsonl``.

    `meta` is written once to ``meta.json`` (config, mechanism, version).
    ``fresh=True`` (a non-resumed run) truncates any previous trace so
    iteration numbers cannot be double-counted by the monitor.
    """

    def __init__(
        self,
        directory: str | Path,
        meta: dict[str, Any] | None = None,
        streams: Iterable[str] = STREAMS,
        fresh: bool = False,
    ) -> None:
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.streams = tuple(streams)
        self._lock = threading.Lock()
        self.counts: Counter[str] = Counter()
        mode = "w" if fresh else "a"
        self._handles = {
            name: (self.dir / f"{name}.jsonl").open(mode, encoding="utf-8")
            for name in self.streams
        }
        payload = {
            "schema_version": SCHEMA_VERSION,
            "created": time.time(),
            "created_iso": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "streams": list(self.streams),
            **(meta or {}),
        }
        (self.dir / "meta.json").write_text(json.dumps(payload, indent=2, default=str))

    # -- writing ------------------------------------------------------
    def write(self, stream: str, record: dict[str, Any]) -> None:
        self.write_many(stream, (record,))

    def write_many(self, stream: str, records: Iterable[dict[str, Any]]) -> None:
        """Append a batch (records must already be JSON-serialisable)."""
        handle = self._handles.get(stream)
        if handle is None:
            raise KeyError(f"unknown trace stream {stream!r}")
        payload = "".join(
            json.dumps(rec, separators=(",", ":"), default=str) + "\n"
            for rec in records
        )
        if not payload:
            return
        with self._lock:
            handle.write(payload)
            handle.flush()
            self.counts[stream] += payload.count("\n")

    # -- lifecycle ----------------------------------------------------
    def flush(self) -> None:
        with self._lock:
            for handle in self._handles.values():
                handle.flush()

    def close(self) -> None:
        with self._lock:
            for handle in self._handles.values():
                try:
                    handle.flush()
                    handle.close()
                except Exception:  # noqa: BLE001 - closing must not raise
                    pass

    def __enter__(self) -> "TraceWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def read_meta(directory: str | Path) -> dict[str, Any]:
    """Trace provenance, or {} when the directory has no trace."""
    path = Path(directory) / "meta.json"
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def trace_path(directory: str | Path, stream: str) -> Path:
    return Path(directory) / f"{stream}.jsonl"
