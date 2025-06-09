"""Synthetic scenario generator scaffold."""

from arc.core.models.scenario import Scenario

__all__: list[str] = ["ScenarioGenerator"]


class ScenarioGenerator:  # noqa: D101
    def generate(self, count: int = 1) -> list[Scenario]:  # noqa: D401
        """Return *count* dummy scenarios (placeholder)."""

        return [Scenario(id=str(i), prompt="TODO") for i in range(count)]
