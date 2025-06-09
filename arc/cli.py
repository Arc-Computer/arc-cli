"""Command-line interface scaffold for the Arc project.

This stub ensures that the `arc` console script defined in pyproject.toml
imports successfully. Additional commands will be added as development
progresses.
"""

from __future__ import annotations

import importlib.metadata as _metadata
import sys

import click


def _get_version() -> str:
    """Return the installed version of *arc-cli* if available.

    When running from a source checkout the package may not yet be
    installed; in that case we fall back to a dev string.
    """
    try:
        return _metadata.version("arc-cli")
    except _metadata.PackageNotFoundError:
        return "0.0.0-dev"


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(_get_version())
def main() -> None:  # pragma: no cover
    """Root *arc* command – currently a no-op scaffold."""


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main()) 