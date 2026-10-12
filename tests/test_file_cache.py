from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from distill_jev.file_cache import FileCache, file_cache


class FileCacheTests(unittest.TestCase):
    def test_creates_missing_cache_and_reuses_it(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "value.pkl"
            calls = 0

            def factory() -> dict[str, int]:
                nonlocal calls
                calls += 1
                return {"answer": 42}

            cached = file_cache(path, factory)

            self.assertEqual(cached(), {"answer": 42})
            self.assertTrue(path.is_file())
            self.assertEqual(cached(), {"answer": 42})
            self.assertEqual(calls, 1)

    def test_cache_can_be_reopened_with_another_factory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "value.pkl"
            _ = FileCache(path, lambda: ["stored"])()

            cached = FileCache(path, lambda: self.fail("factory should not run"))

            self.assertEqual(cached(), ["stored"])
