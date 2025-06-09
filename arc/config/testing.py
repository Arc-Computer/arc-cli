"""Testing environment settings (scaffold)."""

from . import Settings


class TestingSettings(Settings):  # noqa: D101
    debug: bool = True
    test_mode: bool = True
