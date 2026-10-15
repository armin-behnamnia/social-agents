#!/usr/bin/env python
"""Extract rendered messages (+ judge score) for chosen strategy indices.

Offline: reads the warm .llm_cache, makes no LLM calls.

Usage: PYTHONPATH=. python report/extract_messages.py 90,168,58,56
Writes report/messages.json.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

from code.config import GLOBAL_LLM_CONFIG as C
from code.strategy import index_to_strategy
from code.profiles import generate_synthetic_profiles
from code.LLM_module import _PERSUASIVENESS_PROMPT, _NUMBER_RE

HERE = Path(__file__).resolve().parent
CACHE = HERE.parent / ".llm_cache"

RENDER_PROMPT = """You are crafting a persuasive social-media message.

Persuasion strategy:
- Rhetorical appeal: {appeal} (logos = facts/logic, pathos = emotion, ethos = credibility)
- Tone: {tone}
- Emphasis: {emphasis} (evidence = data and studies, personal_story = lived experience, social_proof = what others are doing, risk = dangers of inaction, authority = expert consensus)
- Intensity: {label} ({intensity:.2f}/1.00)

Target audience profile: {profile}

Write ONE short social-media post (maximum 50 words, first person) that
takes an {stance} stance on {topic} and follows the strategy
above as closely as possible. Do not mention the strategy explicitly.
Return only the message."""


def key(prompt, temperature, max_tokens, system=None):
    return hashlib.sha256(json.dumps({
        "model": C.model, "prompt": prompt, "system": system,
        "temperature": temperature, "max_tokens": max_tokens,
    }, sort_keys=True).encode()).hexdigest()


def read(k):
    p = CACHE / f"{k}.json"
    return json.loads(p.read_text())["content"] if p.exists() else None


def render(spec, profile, topic="the policy", stance="opposing"):
    prompt = RENDER_PROMPT.format(
        appeal=spec.appeal, tone=spec.tone, emphasis=spec.emphasis,
        label=spec.intensity_label, intensity=spec.intensity,
        profile=profile, stance=stance, topic=topic)
    return read(key(prompt, 0.0, C.generation_max_tokens))


def score(msg, profile):
    raw = read(key(_PERSUASIVENESS_PROMPT.format(profile=profile, message=msg),
                   C.inference_temperature, C.inference_max_tokens))
    if raw is None:
        return None
    m = _NUMBER_RE.search(raw)
    return min(max(float(m.group()) / 10.0, 0.0), 1.0) if m else 0.5


def main():
    idxs = [int(x) for x in (sys.argv[1] if len(sys.argv) > 1
                             else "90,168,58,56").split(",")]
    profiles = generate_synthetic_profiles(50, seed=1234)
    prof = profiles[0]
    out = []
    for i in idxs:
        spec = index_to_strategy(i)
        msg = render(spec, prof)
        out.append({
            "idx": i,
            "appeal": spec.appeal,
            "tone": spec.tone,
            "emphasis": spec.emphasis,
            "intensity": spec.intensity_label,
            "score": score(msg, prof) if msg else None,
            "message": msg,
        })
    (HERE / "messages.json").write_text(json.dumps(out, indent=1))
    for r in out:
        print(f"{r['idx']:>3} {r['appeal']}/{r['tone']}/{r['emphasis']}/{r['intensity']} score={r['score']}")
    print("wrote report/messages.json")


if __name__ == "__main__":
    main()
