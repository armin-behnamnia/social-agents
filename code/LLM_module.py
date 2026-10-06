"""LLM call layer with retries, caching, and robust output parsing.

Fixes applied vs. the original code:
- Five copy-pasted parameter blocks collapsed into one `call_llm` helper.
- Numeric observer outputs are extracted with a regex and validated to
  [0, 1] (previously raw strings were returned, corrupting metrics).
- Retries with exponential backoff on API/transient failures.
- Deterministic temperature for estimators (was 0.8, injecting noise
  directly into the measured quantity).
- Model / base URL / API key come from config, not hardcoded literals.
- An optional on-disk cache avoids re-paying inference cost on re-runs.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from pathlib import Path

from .config import GLOBAL_LLM_CONFIG

logger = logging.getLogger(__name__)

_NUMBER_RE = re.compile(r"[-+]?\d*\.?\d+")


class LLMError(RuntimeError):
    pass


def _cache_path(cache_dir: Path, key: str) -> Path:
    return cache_dir / f"{key}.json"


def call_llm(
    prompt: str,
    *,
    temperature: float | None = None,
    max_tokens: int | None = None,
    cache: bool = True,
    cache_dir: str | Path = ".llm_cache",
    system_prompt: str | None = None,
) -> str:
    """Single entry point for chat completion with retry + cache."""
    cfg = GLOBAL_LLM_CONFIG

    if temperature is None:
        temperature = cfg.generation_temperature
    if max_tokens is None:
        max_tokens = cfg.generation_max_tokens

    cache_dir_path = Path(cache_dir)
    key = hashlib.sha256(
        json.dumps(
            {
                "model": cfg.model,
                "prompt": prompt,
                "system": system_prompt,
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()

    if cache:
        cache_dir_path.mkdir(parents=True, exist_ok=True)
        cached = _cache_path(cache_dir_path, key)
        if cached.exists():
            try:
                return json.loads(cached.read_text())["content"]
            except (json.JSONDecodeError, KeyError):
                logger.warning("corrupt cache entry %s; ignoring", cached)

    # Imported lazily so tests / analysis can run without the openai package.
    try:
        from openai import OpenAI
    except ImportError as exc:  # pragma: no cover
        raise LLMError(
            "openai package not installed; install requirements.txt"
        ) from exc

    client = OpenAI(
        base_url=cfg.base_url,
        api_key=cfg.api_key,
        timeout=cfg.timeout_seconds,
    )

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    last_error: Exception | None = None
    for attempt in range(1, cfg.max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=cfg.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            content = response.choices[0].message.content
            if content is None or not str(content).strip():
                raise LLMError("LLM returned empty content")
            content = str(content).strip()
            if cache:
                _cache_path(cache_dir_path, key).write_text(
                    json.dumps({"content": content})
                )
            return content
        except Exception as exc:  # noqa: BLE001 - transient API errors
            last_error = exc
            wait = cfg.retry_backoff_seconds * (2 ** (attempt - 1))
            logger.warning(
                "LLM call attempt %d/%d failed (%s); retrying in %.1fs",
                attempt,
                cfg.max_retries,
                exc,
                wait,
            )
            time.sleep(wait)

    raise LLMError(f"LLM call failed after {cfg.max_retries} attempts") from last_error


def parse_stance_shift(raw: str) -> float:
    """Extract a validated stance-shift value in [0, 1] from an LLM reply.

    Returns 0.0 when nothing numeric can be extracted (previously raw
    strings like "The shift is 0.3." flowed into metric computation).
    """
    if raw is None:
        return 0.0
    matches = _NUMBER_RE.findall(str(raw))
    for match in reversed(matches):  # last number is most likely the answer
        try:
            value = float(match)
        except ValueError:
            continue
        if 0.0 <= value <= 1.0:
            return value
        # allow e.g. "0.8 out of 1" style answers already covered; clamp others
        if value > 1.0:
            continue
    # fall back to first numeric token clamped
    for match in matches:
        try:
            value = float(match)
        except ValueError:
            continue
        return min(max(value, 0.0), 1.0)
    return 0.0


# ----------------------------------------------------------------------
# Prompts (kept close to the paper's Prompt 1-5 definitions)
# ----------------------------------------------------------------------

PROFILE_GENERATION_PROMPT = """Generate user profiles for {n_users} users as a 50-word unique description,
ages of these users follow Gaussian distribution,
gender is half and half.
Listed your response by user id.

Based on this example:
User ID 15: Benjamin is a 48-year-old male who is a history enthusiast.
He enjoys reading historical books and visiting museums."""


def generate_profile(n_users: int = 50) -> str:
    return call_llm(
        PROFILE_GENERATION_PROMPT.format(n_users=n_users),
        temperature=GLOBAL_LLM_CONFIG.generation_temperature,
        max_tokens=GLOBAL_LLM_CONFIG.profile_max_tokens,
    )


def generate_information(received: str, profile: str) -> str:
    prompt = f"""You are a social media user with the following profile: {profile}

You have received the following information: {received}

Generate one new piece of information that conveys or spreads the received information.

The generated message should reflect:
- your personal profile,
- the received information,

Do not follow a fixed rewriting rule. Infer a natural way to express the information.

Write in first person. Maximum 50 words."""
    return call_llm(prompt)


def update_profile(received: str, profile: str) -> str:
    prompt = f"""You are a social media user with the following profile: {profile}

You have received the following information: {received}

Generate one piece of information as the new profile that conveys or spreads the received information.

The generated profile should reflect:
- your current personal profile,
- the received information,

Do not follow a fixed rewriting rule. Infer a natural way to express the information.

Write in first person. Maximum 50 words."""
    return call_llm(prompt)


def generate_response_message(post: str, profile: str, delta_stance: float) -> str:
    prompt = f"""You are given an Original Post that has influenced a target user.

The target user has **not fully accepted the Original Post**. Instead, based on their profile, they have unconsciously internalized only the part of its viewpoint that is consistent with them, resulting in a stance change of {delta_stance:.4f} toward the Original Post.

Identify and express the part of the Original Post that the target user has internalized. The generated message should preserve the original viewpoint but reflect **only the portion accepted by this user**, corresponding to the specified stance change.

Original Post:
{post}

Target Profile:
{profile}

Stance Change (delta):
{delta_stance:.4f}

Generate only the internalized message."""
    return call_llm(prompt)


def _infer_shift_prompt_body(received_message: str) -> str:
    return f"""The user may accept only part of the message. Do not assume that receiving or responding to the message means full acceptance.

Stance shift is a value between 0 and 1 representing the degree of change toward the Received Message.

- 0 = no observable shift toward the message
- 0.25 = small shift
- 0.5 = moderate shift
- 0.75 = substantial shift
- 1 = strong shift toward the message

Estimate the shift caused by the Received Message, not the general sentiment of the New Post or Profile.

Received Message:
{received_message}

"""


def inference_stance(
    received_message: str, new_post: str, previous_profile: str, updated_profile: str
) -> float:
    prompt = f"""You are an observer estimating how much a user's stance shifted toward a received message.

The user first had a Previous Profile. After receiving the Received Message, the user generated a New Post and their profile changed to the Updated Profile.

Estimate how much the Received Message changed the user's stance toward its viewpoint. Use the New Post and the change from the Previous Profile to the Updated Profile as observable evidence.

{_infer_shift_prompt_body(received_message)}
Previous Profile:
{previous_profile}

New Post:
{new_post}

Updated Profile:
{updated_profile}

Return only one numerical value in [0, 1]."""
    raw = call_llm(
        prompt,
        temperature=GLOBAL_LLM_CONFIG.inference_temperature,
        max_tokens=GLOBAL_LLM_CONFIG.inference_max_tokens,
    )
    return parse_stance_shift(raw)


def inference_stance_post(received_message: str, new_post: str) -> float:
    prompt = f"""You are an observer estimating how much a user's stance shifted toward a received message.

After receiving the Received Message, the user generated a New Post in response.

Estimate how much the Received Message changed the user's stance toward its viewpoint, using the New Post as observable evidence of the user's response.

{_infer_shift_prompt_body(received_message)}
New Post:
{new_post}

Return only one numerical value in [0, 1]."""
    raw = call_llm(
        prompt,
        temperature=GLOBAL_LLM_CONFIG.inference_temperature,
        max_tokens=GLOBAL_LLM_CONFIG.inference_max_tokens,
    )
    return parse_stance_shift(raw)


def inference_stance_profile(
    received_message: str, previous_profile: str, updated_profile: str
) -> float:
    prompt = f"""You are an observer estimating how much a user's stance shifted toward a received message.

The user first had a Previous Profile. After receiving the Received Message, the user's profile changed to the Updated Profile.

Estimate how much the Received Message changed the user's stance toward its viewpoint, using the change from the Previous Profile to the Updated Profile as observable evidence.

{_infer_shift_prompt_body(received_message)}
Previous Profile:
{previous_profile}

Updated Profile:
{updated_profile}

Return only one numerical value in [0, 1]."""
    raw = call_llm(
        prompt,
        temperature=GLOBAL_LLM_CONFIG.inference_temperature,
        max_tokens=GLOBAL_LLM_CONFIG.inference_max_tokens,
    )
    return parse_stance_shift(raw)


if __name__ == "__main__":
    print(generate_profile())


# ----------------------------------------------------------------------
# Strategy-conditioned persuasion rendering (RL approach 2)
# ----------------------------------------------------------------------

def render_strategy_message(
    strategy_spec,
    target_profile: str,
    topic: str = "the policy",
    stance_direction: str = "opposing",
) -> str:
    """Render a StrategySpec into a persuasion message via the LLM.

    The LLM is frozen (not trained); it acts as a decoder from the
    strategy space to natural language. Deterministic temperature so
    the RL signal reflects the policy's action, not sampling noise.
    """
    prompt = f"""You are crafting a persuasive social-media message.

Persuasion strategy:
- Rhetorical appeal: {strategy_spec.appeal} (logos = facts/logic, pathos = emotion, ethos = credibility)
- Tone: {strategy_spec.tone}
- Emphasis: {strategy_spec.emphasis} (evidence = data and studies, personal_story = lived experience, social_proof = what others are doing, risk = dangers of inaction, authority = expert consensus)
- Intensity: {strategy_spec.intensity_label} ({strategy_spec.intensity:.2f}/1.00)

Target audience profile: {target_profile}

Write ONE short social-media post (maximum 50 words, first person) that
takes an {stance_direction} stance on {topic} and follows the strategy
above as closely as possible. Do not mention the strategy explicitly.
Return only the message."""
    return call_llm(
        prompt,
        temperature=0.0,
        max_tokens=GLOBAL_LLM_CONFIG.generation_max_tokens,
    )
