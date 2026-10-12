import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import cast
from unittest.mock import patch

import polars as pl

from distill_jev.datasets.deduplication import deduplicate_chats


class DeduplicationTests(unittest.TestCase):
    def test_deduplicates_exact_content_and_keeps_first_record(self) -> None:
        with TemporaryDirectory() as directory:
            source = Path(directory) / "source.jsonl"
            destination = Path(directory) / "nested" / "result.jsonl"
            rows = [
                {"id": 1, "content": "hello 🌍", "metadata": {"server": "one"}},
                {"id": 2, "content": "hello 🌍", "metadata": {"server": "two"}},
                {"id": 3, "content": "Hello 🌍", "metadata": {"server": "three"}},
                {"id": 4, "content": "", "metadata": {"server": "four"}},
                {"id": 5, "content": "", "metadata": {"server": "five"}},
            ]
            original = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
            _ = source.write_text(original, encoding="utf-8")

            self.assertEqual(deduplicate_chats(source, destination), destination)

            actual = [cast(dict[str, object], json.loads(line)) for line in destination.read_text().splitlines()]
            self.assertEqual({row["id"] for row in actual}, {1, 3, 4})
            self.assertEqual({row["content"] for row in actual}, {"hello 🌍", "Hello 🌍", ""})
            self.assertIn(rows[0], actual)
            self.assertEqual(source.read_text(encoding="utf-8"), original)

    def test_retains_fields_introduced_after_schema_inference_sample(self) -> None:
        with TemporaryDirectory() as directory:
            source = Path(directory) / "source.jsonl"
            destination = Path(directory) / "result.jsonl"
            rows: list[dict[str, object]] = [{"content": str(index)} for index in range(101)]
            rows.append({"content": "last", "extra": {"server": "late"}})
            _ = source.write_text("".join(json.dumps(row) + "\n" for row in rows))

            _ = deduplicate_chats(source, destination)

            actual = [cast(dict[str, object], json.loads(line)) for line in destination.read_text().splitlines()]
            self.assertIn(rows[-1], actual)
            self.assertEqual(len(actual), len(rows))

    def test_empty_input_replaces_existing_output(self) -> None:
        with TemporaryDirectory() as directory:
            source = Path(directory) / "source.jsonl"
            destination = Path(directory) / "result.jsonl"
            source.touch()
            _ = destination.write_text("old output")

            _ = deduplicate_chats(source, destination)

            self.assertEqual(destination.read_bytes(), b"")

    def test_rejects_same_path_without_changing_source(self) -> None:
        with TemporaryDirectory() as directory:
            source = Path(directory) / "source.jsonl"
            original = '{"content":"hello"}\n'
            _ = source.write_text(original)

            with self.assertRaisesRegex(ValueError, "source and destination must differ"):
                _ = deduplicate_chats(source, source)

            self.assertEqual(source.read_text(), original)

    def test_invalid_input_preserves_existing_destination(self) -> None:
        for invalid in ['{"content":5}', '{"content":null}', '{}', '[]', '"text"', 'not json', '']:
            with self.subTest(invalid=invalid), TemporaryDirectory() as directory:
                source = Path(directory) / "source.jsonl"
                destination = Path(directory) / "result.jsonl"
                original = '{"content":"valid"}\n' + invalid + "\n"
                _ = source.write_text(original)
                _ = destination.write_text("old output")

                with self.assertRaises(ValueError):
                    _ = deduplicate_chats(source, destination)

                self.assertEqual(destination.read_text(), "old output")
                self.assertEqual(source.read_text(), original)

    def test_polars_write_failure_preserves_existing_destination(self) -> None:
        with TemporaryDirectory() as directory:
            source = Path(directory) / "source.jsonl"
            destination = Path(directory) / "result.jsonl"
            _ = source.write_text('{"content":"hello"}\n')
            _ = destination.write_text("old output")

            with patch.object(pl.LazyFrame, "sink_ndjson", side_effect=RuntimeError("write failed")):
                with self.assertRaisesRegex(RuntimeError, "write failed"):
                    _ = deduplicate_chats(source, destination)

            self.assertEqual(destination.read_text(), "old output")
            self.assertEqual(list(Path(directory).glob("deduplicate-*")), [])
