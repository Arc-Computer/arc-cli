"""Atomic file operations utilities."""

import json
import tempfile
from pathlib import Path
from typing import Any, Union


def atomic_write_json(filepath: Union[str, Path], data: Any, **json_kwargs) -> None:
    """
    Write JSON data to a file atomically.
    
    This prevents partial writes and corruption by writing to a temporary
    file first, then atomically renaming it to the target path.
    
    Args:
        filepath: Target file path
        data: Data to serialize to JSON
        **json_kwargs: Additional arguments passed to json.dump()
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    # Create temp file in same directory to ensure same filesystem
    with tempfile.NamedTemporaryFile(
        mode='w',
        dir=filepath.parent,
        prefix=f".{filepath.name}.",
        suffix='.tmp',
        delete=False
    ) as tmp_file:
        tmp_path = Path(tmp_file.name)
        try:
            # Write data to temp file
            json.dump(data, tmp_file, **json_kwargs)
            tmp_file.flush()
            # Ensure data is written to disk
            tmp_file.close()
            
            # Atomic rename (POSIX guarantees atomicity)
            tmp_path.replace(filepath)
        except Exception:
            # Clean up temp file on error
            try:
                tmp_path.unlink()
            except OSError:
                pass
            raise


def atomic_write_text(filepath: Union[str, Path], content: str) -> None:
    """
    Write text content to a file atomically.
    
    Args:
        filepath: Target file path
        content: Text content to write
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with tempfile.NamedTemporaryFile(
        mode='w',
        dir=filepath.parent,
        prefix=f".{filepath.name}.",
        suffix='.tmp',
        delete=False,
        encoding='utf-8'
    ) as tmp_file:
        tmp_path = Path(tmp_file.name)
        try:
            tmp_file.write(content)
            tmp_file.flush()
            tmp_file.close()
            
            tmp_path.replace(filepath)
        except Exception:
            try:
                tmp_path.unlink()
            except OSError:
                pass
            raise