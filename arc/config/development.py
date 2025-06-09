"""Development environment settings (scaffold)."""

from . import Settings


class DevelopmentSettings(Settings):  # noqa: D101
    debug: bool = True
