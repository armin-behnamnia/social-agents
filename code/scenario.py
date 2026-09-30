"""Scenario definition replacing the missing Melodie-based scenario."""

from __future__ import annotations

from dataclasses import dataclass, field

from .config import ScenarioConfig


@dataclass
class LLMScenario:
    """Lightweight scenario object with attribute access used by agents."""

    agent_num: int = 50
    inference_state_min: float = -1.0
    inference_state_max: float = 1.0
    stance_influence_rate: float = 0.4
    stance_threshold: float = 0.5
    adoption_threshold: float = 0.0
    alpha_profile: float = 0.25
    alpha_content: float = 0.25
    alpha_stance: float = 0.25
    alpha_degree: float = 0.25
    seed: int | None = 42
    use_post_evidence: bool = True
    use_profile_evidence: bool = True
    output_dir: str = "output"
    rng_seed: int | None = None

    @classmethod
    def from_config(cls, cfg: ScenarioConfig | None = None) -> "LLMScenario":
        cfg = cfg or ScenarioConfig()
        return cls(
            agent_num=cfg.agent_num,
            inference_state_min=cfg.inference_state_min,
            inference_state_max=cfg.inference_state_max,
            stance_influence_rate=cfg.stance_influence_rate,
            stance_threshold=cfg.stance_threshold,
            adoption_threshold=cfg.adoption_threshold,
            alpha_profile=cfg.alpha_profile,
            alpha_content=cfg.alpha_content,
            alpha_stance=cfg.alpha_stance,
            alpha_degree=cfg.alpha_degree,
            seed=cfg.seed,
            use_post_evidence=cfg.use_post_evidence,
            use_profile_evidence=cfg.use_profile_evidence,
            output_dir=cfg.output_dir,
            rng_seed=cfg.seed,
        )
