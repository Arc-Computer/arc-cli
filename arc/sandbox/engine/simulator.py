"""Core simulator placeholder."""

from __future__ import annotations

from arc.core.models.scenario import Scenario

__all__: list[str] = ["Simulator"]


class Simulator:  # noqa: D101
    async def run(self, scenario: Scenario):  # noqa: D401
        """Execute *scenario* (not yet implemented)."""

        raise NotImplementedError("Simulator.run() not implemented")
