"""Persistent, file-backed caching for values that can be pickled."""

from __future__ import annotations

import os
import pickle
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import cast


class FileCache[T]:
    """Cache the result of a zero-argument factory in a local file.

    The factory is only evaluated when ``path`` does not exist. Values are
    serialized with :mod:`pickle`, so only load cache files from trusted
    locations.
    """

    path: Path
    factory: Callable[[], T]

    def __init__(self, path: str | Path, factory: Callable[[], T]) -> None:
        self.path = Path(path)
        self.factory = factory

    def __call__(self) -> T:
        """Return the cached value, creating and persisting it if needed."""
        if self.path.is_file():
            with self.path.open("rb") as cache_file:
                return cast(T, pickle.load(cache_file))

        value = self.factory()
        self._write(value)
        return value

    def _write(self, value: T) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_path = tempfile.mkstemp(
            dir=self.path.parent,
            prefix=f".{self.path.name}.",
            suffix=".tmp",
        )
        try:
            with os.fdopen(descriptor, "wb") as cache_file:
                pickle.dump(value, cache_file, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(temporary_path, self.path)
        except BaseException:
            Path(temporary_path).unlink(missing_ok=True)
            raise


def file_cache[T](path: str | Path, factory: Callable[[], T]) -> FileCache[T]:
    """Create a callable cache for ``factory`` stored at ``path``.

    Example:
        settings = file_cache(".cache/settings.pkl", load_settings)
        value = settings()
    """
    return FileCache(path, factory)
