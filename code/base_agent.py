"""BaseAgent: minimal stand-in for Melodie's NetworkAgent.

The original code inherited from `Melodie.NetworkAgent` but only used
`setup()` and attribute storage, so a self-contained base class keeps
the simulation runnable while preserving the structure.
"""

from __future__ import annotations

from typing import Any


class BaseAgent:
    def __init__(self, agent_id: int):
        self.id = agent_id
        self.category = 0
        self.setup()

    def setup(self) -> None:
        pass

    def __repr__(self) -> str:
        attrs = {k: v for k, v in self.__dict__.items() if not k.startswith("_")}
        return f"{type(self).__name__}({attrs})"

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


class NetworkAgent(BaseAgent):
    def __init__(self, agent_id: int, neighbors: set[int] | None = None):
        self._neighbors: set[int] = set(neighbors or [])
        super().__init__(agent_id)

    @property
    def neighbors(self) -> set[int]:
        return self._neighbors

    def add_neighbor(self, agent_id: int) -> None:
        self._neighbors.add(agent_id)
