"""Production environment settings (scaffold)."""

from . import Settings


class ProductionSettings(Settings):  # noqa: D101
    debug: bool = False
