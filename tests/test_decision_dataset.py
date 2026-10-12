"""CPU-only checks for sharding and resume identity."""

import json
import unittest
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from distill_jev.models.decision.perplexity.dataset import dataset_manifest, split_dataset

QUESTION = {"type": "choice", "criteria": {"b": "B", "a": "A"}}


class DatasetTests(unittest.TestCase):
    def test_shards_preserve_original_bytes_metadata_and_order(self):
        rows = [{"content": f"héllo {i}", "moderation": {"OK": 0.3},
                 "sampling": {"source_row_index": i}} for i in range(259)]
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

    def test_invalid_text_existing_scores_and_shard_size(self):
        with TemporaryDirectory() as temp:
            source = Path(temp) / "input"
            for row in ({"content": 42}, {"content": "hello", "decision": {}}):
                source.write_text(json.dumps(row))
                with self.assertRaisesRegex(ValueError, "Row 1"):
                    split_dataset(source, temp)
            with self.assertRaises(ValueError):
                split_dataset(source, temp, rows_per_shard=0)

    def test_custom_field_and_existing_files(self):
        with TemporaryDirectory() as temp:
            source = Path(temp) / "input"
            source.write_text('{"message":"hello","images":["source metadata"]}\n')
            shards, _, _ = split_dataset(source, temp, text_field="message")
            self.assertEqual(shards[0].read_bytes(), source.read_bytes())
            with self.assertRaises(FileExistsError):
                split_dataset(source, temp, text_field="message")

    def test_candidate_order_is_semantic_and_question_is_not_mutated(self):
        def manifest(question):
            return dataset_manifest("checksum", 1, question=question)

        self.assertEqual(manifest(QUESTION), manifest(dict(reversed(list(QUESTION.items())))))
        self.assertNotEqual(manifest(QUESTION), manifest({**QUESTION, "criteria": {"a": "A", "b": "B"}}))
        self.assertEqual(json.loads(manifest(QUESTION))["question"]["criteria"], [["b", "B"], ["a", "A"]])
        self.assertEqual(QUESTION["criteria"], {"b": "B", "a": "A"})
        self.assertNotEqual(manifest({"type": "score", "criteria": ["low", "high"]}),
                            manifest({"type": "score", "criteria": ["high", "low"]}))

    def test_identity_covers_input_and_inference_settings(self):
        baseline = dataset_manifest("checksum", 1, question=QUESTION)
        for kwargs in ({"text_field": "message"}, {"batch_size": 16}, {"rows_per_shard": 64}):
            self.assertNotEqual(baseline, dataset_manifest("checksum", 1, question=QUESTION, **kwargs))
        for checksum, count in (("different", 1), ("checksum", 2)):
            self.assertNotEqual(baseline, dataset_manifest(checksum, count, question=QUESTION))
        for name in ("MODEL_REVISION", "PIPELINE_VERSION", "NUMERICAL_POLICY", "PROCESSING_WINDOW"):
            with patch(f"distill_jev.models.decision.perplexity.dataset.{name}", "changed"):
                self.assertNotEqual(baseline, dataset_manifest("checksum", 1, question=QUESTION))
