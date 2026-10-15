"""CPU-only checks for sharding and resume identity."""

import json
import unittest
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from themis.models.decision.perplexity.dataset import (
    dataset_manifest,
    prepare_dataset,
    split_dataset,
)

QUESTION = {"type": "choice", "criteria": {"b": "B", "a": "A"}}


class DatasetTests(unittest.TestCase):
    def test_shards_preserve_original_bytes_metadata_and_order(self):
        rows = [{"state": f"héllo {i}", "question": QUESTION,
                 "source": {"content": f"héllo {i}", "moderation": {"OK": 0.3},
                            "sampling": {"source_row_index": i}}} for i in range(259)]
        raw = b"".join((json.dumps(row, ensure_ascii=False) +
                        ("\r\n" if i < 258 else "")).encode() for i, row in enumerate(rows))
        with TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "input"
            source.write_bytes(raw)
            with patch.object(Path, "read_bytes", side_effect=AssertionError("not streaming")):
                shards, checksum, total = split_dataset(source, root, rows_per_shard=128)
            self.assertEqual([path.name for path in shards],
                             ["000000.jsonl", "000001.jsonl", "000002.jsonl"])
            self.assertEqual([len(path.read_bytes().splitlines()) for path in shards], [128, 128, 3])
            self.assertEqual(b"".join(path.read_bytes() for path in shards), raw)
            self.assertEqual(checksum, sha256(raw).hexdigest())
            self.assertEqual(total, 259)

    def test_empty_input(self):
        with TemporaryDirectory() as temp:
            source = Path(temp) / "input"
            source.touch()
            self.assertEqual(split_dataset(source, temp), ([], sha256(b"").hexdigest(), 0))

    def test_invalid_native_rows_existing_scores_and_shard_size(self):
        with TemporaryDirectory() as temp:
            source = Path(temp) / "input"
            for row in ({"state": "hello"}, {"state": "hello", "question": {"type": "noul"}, "decision": {}}):
                source.write_text(json.dumps(row))
                with self.assertRaisesRegex(ValueError, "Row 1"):
                    split_dataset(source, temp)
            with self.assertRaises(ValueError):
                split_dataset(source, temp, rows_per_shard=0)

    def test_preparation_preserves_source_and_can_include_native_fields(self):
        with TemporaryDirectory() as temp:
            source = Path(temp) / "input"
            prepared = Path(temp) / "prepared"
            record = {"message": "hello", "images": ["data:image/png;base64,xxx"], "id": 3}
            source.write_text(json.dumps(record) + "\n")
            self.assertEqual(prepare_dataset(source, prepared, question=QUESTION,
                                             text_field="message", native_fields=("images",)), prepared)
            row = json.loads(prepared.read_text())
            self.assertEqual(row["state"], "hello")
            self.assertEqual(row["question"], QUESTION)
            self.assertEqual(row["images"], record["images"])
            self.assertEqual(row["source"], record)

    def test_candidate_order_is_preserved_by_prepared_input_bytes(self):
        with TemporaryDirectory() as temp:
            source = Path(temp) / "source"
            source.write_text('{"content":"hello"}\n')
            first, second = Path(temp) / "first", Path(temp) / "second"
            prepare_dataset(source, first, question=QUESTION)
            prepare_dataset(source, second, question={**QUESTION, "criteria": {"a": "A", "b": "B"}})
            self.assertNotEqual(first.read_bytes(), second.read_bytes())
            self.assertEqual(QUESTION["criteria"], {"b": "B", "a": "A"})

    def test_identity_covers_input_and_inference_settings(self):
        baseline = dataset_manifest("checksum", 1)
        for kwargs in ({"batch_size": 16}, {"rows_per_shard": 64}):
            self.assertNotEqual(baseline, dataset_manifest("checksum", 1, **kwargs))
        for checksum, count in (("different", 1), ("checksum", 2)):
            self.assertNotEqual(baseline, dataset_manifest(checksum, count))
        for name in ("MODEL_REVISION", "PIPELINE_VERSION", "NUMERICAL_POLICY", "PROCESSING_WINDOW"):
            with patch(f"themis.models.decision.perplexity.dataset.{name}", "changed"):
                self.assertNotEqual(baseline, dataset_manifest("checksum", 1))
