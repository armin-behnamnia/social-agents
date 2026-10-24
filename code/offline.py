"""Offline mock backend for LLM_module.

Patches the module's LLM-backed functions with deterministic canned
responses so the simulation, tests, and RL rollouts run without a live
LLM server. Useful for:
- smoke-testing the full pipeline offline,
- fast RL training loops before wiring in the real generator,
- CI.

Not scientifically meaningful: message content is fixed. The stance
dynamics (which do not depend on LLM output) remain fully functional.
"""

from __future__ import annotations

import hashlib

from . import LLM_module


def _hash01(*parts: str) -> float:
    digest = hashlib.md5("|".join(parts).encode()).digest()
    return int.from_bytes(digest[:4], "little") / 2**32


def install() -> None:
    LLM_module.generate_profile = lambda n_users=50: "\n".join(
        f"User ID {i}: Mock user {i} is a {20 + i % 50}-year-old person "
        f"interested in topic {i % 5}." 
        for i in range(n_users)
    )
    LLM_module.generate_information = (
        lambda received, profile: f"Speaking as ({profile[:30]}...), "
        f"I think this matters: {received[:60]}"
    )
    LLM_module.update_profile = (
        lambda received, profile: f"{profile[:40]} | updated view: {received[:40]}"
    )
    LLM_module.generate_response_message = (
        lambda post, profile, delta_stance: f"Partially internalized "
        f"({delta_stance:.2f}) view of: {post[:60]}"
    )
    LLM_module.inference_stance_post = (
        lambda received_message, new_post: _hash01("post", received_message, new_post)
    )
    LLM_module.inference_stance_profile = (
        lambda received_message, previous_profile, updated_profile: _hash01(
            "profile", received_message, previous_profile, updated_profile
        )
    )
    LLM_module.inference_stance = (
        lambda received_message, new_post, previous_profile, updated_profile: _hash01(
            "combined", received_message, new_post, previous_profile, updated_profile
        )
    )
    def _mock_render(strategy_spec, target_profile, topic="the policy",
                     stance_direction="opposing"):
        if hasattr(strategy_spec, "angle"):  # v2 space
            tag = (f"{strategy_spec.angle}/{strategy_spec.evidence}"
                   f"/{strategy_spec.address}")
        else:  # v1 grid
            tag = (f"{strategy_spec.appeal}/{strategy_spec.tone}"
                   f"/{strategy_spec.emphasis}/{strategy_spec.intensity:.2f}")
        return f"[{tag}] message aimed at: {target_profile[:40]}"

    LLM_module.render_strategy_message = _mock_render
    LLM_module.judge_persuasiveness = (
        lambda message, target_profile: _hash01(
            "judge", message, target_profile
        )
    )
    # Melodie-mechanism accept gate: mostly-accepting deterministic stub.
    LLM_module.accept_message = (
        lambda post, profile, history=None: (
            "ACCEPT"
            if _hash01("accept", str(post), str(profile)) < 0.7
            else "REJECT"
        )
    )


def uninstall() -> None:
    """Restore the real implementations (best effort)."""
    import importlib

    importlib.reload(LLM_module)
