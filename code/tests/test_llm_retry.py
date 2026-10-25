"""Tests for the transient-failure policy in code/LLM_module.py.

This policy is not academic: a campaign that is hours into training was
killed ~2 minutes after a parallel relaunch because the shared gateway
answered 503 for longer than the old 3-attempt / ~10 s retry window. So
the contract these tests pin is:

- backoff grows exponentially, so an early retry is cheap;
- it is **capped**, so a large retry budget never sleeps for minutes;
- it is **jittered** multiplicative in [0.5, 1.5), so hundreds of workers
  that failed together do not retry together (correlated retries are what
  sustain a 503 burst);
- a call that keeps failing still gives up after ``max_retries`` and
  surfaces the last error rather than looping.
"""

from __future__ import annotations

import dataclasses

import pytest

from code import LLM_module as llm


def _cfg(**overrides) -> llm.LLMConfig:
    """The global config with a fast retry policy for the unit test."""
    base = dict(
        max_retries=3,
        retry_backoff_seconds=2.0,
        retry_backoff_max_seconds=10.0,
    )
    base.update(overrides)
    return dataclasses.replace(llm.GLOBAL_LLM_CONFIG, **base)


# --------------------------------------------------------------------------
# retry_wait: the backoff schedule itself
# --------------------------------------------------------------------------


class TestRetryWaitSchedule:
    def test_grows_exponentially_until_the_cap(self):
        cfg = _cfg(retry_backoff_seconds=2.0, retry_backoff_max_seconds=100.0)
        # the jittered value is base * [0.5, 1.5); compare the bounds
        for attempt, base in ((1, 2.0), (2, 4.0), (3, 8.0), (4, 16.0)):
            wait = llm.retry_wait(cfg, attempt)
            assert base * 0.5 <= wait < base * 1.5, attempt

    def test_is_capped(self):
        cfg = _cfg(retry_backoff_seconds=2.0, retry_backoff_max_seconds=10.0)
        # raw backoff would be 16 s, 32 s, ... -- the cap holds it at 10 s
        for attempt in (4, 5, 8, 32):
            wait = llm.retry_wait(cfg, attempt)
            assert 10.0 * 0.5 <= wait < 10.0 * 1.5, attempt

    def test_jitter_spans_the_interval(self, monkeypatch):
        cfg = _cfg(retry_backoff_seconds=2.0, retry_backoff_max_seconds=100.0)
        monkeypatch.setattr(llm.random, "random", lambda: 0.0)
        assert llm.retry_wait(cfg, 3) == pytest.approx(8.0 * 0.5)
        monkeypatch.setattr(llm.random, "random", lambda: 0.999)
        assert llm.retry_wait(cfg, 3) == pytest.approx(8.0 * 1.499)

    def test_never_exceeds_the_cap_times_jitter(self):
        cfg = _cfg(retry_backoff_seconds=1.5, retry_backoff_max_seconds=30.0)
        worst = max(llm.retry_wait(cfg, a) for a in range(1, 40))
        assert worst < 30.0 * 1.5, "a long retry budget must not sleep for minutes"


# --------------------------------------------------------------------------
# call_llm: the loop that uses the schedule
# --------------------------------------------------------------------------


class _Completions:
    def __init__(self, owner):
        self._owner = owner

    def create(self, **_kwargs):
        owner = self._owner
        owner.calls += 1
        if owner.calls <= owner.failures:
            raise RuntimeError(f"503 Service Unavailable (#{owner.calls})")
        message = type("M", (), {"content": owner.content})()
        choice = type("C", (), {"message": message})()
        return type("R", (), {"choices": [choice]})()


class _FlakyClient:
    """Fails the first ``failures`` calls, then answers ``content``."""

    def __init__(self, failures: int, content: str = "ANSWER"):
        self.failures = failures
        self.content = content
        self.calls = 0
        self.chat = type("Chat", (), {"completions": _Completions(self)})()


@pytest.fixture()
def sleeps(monkeypatch):
    """Record (and skip) the backoff sleeps."""
    seen: list[float] = []
    monkeypatch.setattr(llm.time, "sleep", seen.append)
    return seen


class TestCallLlmRetryLoop:
    def test_transient_failures_are_retried_then_succeed(
            self, monkeypatch, sleeps):
        cfg = _cfg()
        client = _FlakyClient(failures=2, content="RECOVERED")
        monkeypatch.setattr(llm, "GLOBAL_LLM_CONFIG", cfg)
        monkeypatch.setattr(llm, "_get_client", lambda: client)

        out = llm.call_llm("prompt", temperature=0.0, max_tokens=8,
                           cache=False)

        assert out == "RECOVERED"
        assert client.calls == 3, "two failures then one success"
        assert len(sleeps) == 2, "sleep only between attempts"
        for first, second in zip(sleeps, sleeps[1:]):
            assert first < second, "backoff must be increasing"

    def test_gives_up_after_max_retries(self, monkeypatch, sleeps):
        cfg = _cfg(max_retries=3)
        client = _FlakyClient(failures=99)
        monkeypatch.setattr(llm, "GLOBAL_LLM_CONFIG", cfg)
        monkeypatch.setattr(llm, "_get_client", lambda: client)

        with pytest.raises(llm.LLMError) as excinfo:
            llm.call_llm("prompt", temperature=0.0, max_tokens=8, cache=False)

        assert client.calls == 3
        assert "after 3 attempts" in str(excinfo.value)
        # the last underlying error is chained for diagnosis
        assert isinstance(excinfo.value.__cause__, RuntimeError)
        assert "503" in str(excinfo.value.__cause__)
        assert len(sleeps) == 2, "no pointless sleep after the last attempt"

    def test_a_larger_budget_actually_helps(self, monkeypatch, sleeps):
        """The campaign setting: a burst longer than 3 attempts survives."""
        cfg = _cfg(max_retries=6)
        client = _FlakyClient(failures=5, content="SURVIVED")
        monkeypatch.setattr(llm, "GLOBAL_LLM_CONFIG", cfg)
        monkeypatch.setattr(llm, "_get_client", lambda: client)

        assert llm.call_llm("prompt", temperature=0.0, max_tokens=8,
                            cache=False) == "SURVIVED"
        assert client.calls == 6
        assert len(sleeps) == 5

    def test_empty_content_is_treated_as_a_failure(self, monkeypatch, sleeps):
        """An empty reply used to be accepted (and poisoned downstream)."""
        cfg = _cfg(max_retries=2)
        client = _FlakyClient(failures=0, content="   ")
        monkeypatch.setattr(llm, "GLOBAL_LLM_CONFIG", cfg)
        monkeypatch.setattr(llm, "_get_client", lambda: client)

        with pytest.raises(llm.LLMError):
            llm.call_llm("prompt", temperature=0.0, max_tokens=8, cache=False)
        assert client.calls == 2
