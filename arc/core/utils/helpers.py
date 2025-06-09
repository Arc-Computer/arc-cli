"""Miscellaneous helper utilities (scaffold)."""

from __future__ import annotations

__all__: list[str] = ["chunk"]

def chunk(seq, size):  # type: ignore[typing-arg-types]
    """Yield successive *size*-sized chunks from *seq* (scaffold)."""

    for i in range(0, len(seq), size):
        yield seq[i : i + size]
