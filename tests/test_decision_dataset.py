"""CPU-only dataset preparation and resumable identity contracts."""

import json
import os
import unittest
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from distill_jev.models.decision.perplexity.dataset import (
    DatasetSettings, dataset_manifest, decision_input, prepare_dataset, validate_dataset_paths,
)
from distill_jev.models.decision.perplexity.questions import validate_question

QUESTION = {"type": "choice", "instructions": "Pick one", "criteria": {"b": "B", "a": "A"}}


class DatasetPreparationTests(unittest.TestCase):
    def test_streaming_shards_hash_positions_metadata_and_order(self):
        records = [{"content": f"héllo {i}", "moderation": {"safe": 0.3},
                    "sampling": {"position": i}, "decision_state": "untouched"}
                   for i in range(259)]
        raw = b"".join(("  " + json.dumps(record, ensure_ascii=False) +
                        ("\r\n" if i < 258 else "")).encode()
                       for i, record in enumerate(records))
        with TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "input.jsonl"
            source.write_bytes(raw)
            scratch = root / "scratch"
            scratch.mkdir()
            output = root / "existing.jsonl"
            output.write_bytes(b"previous scored output")
            # No whole-file read is allowed in the streaming preparation path.
            with patch.object(Path, "read_bytes", side_effect=AssertionError("not streaming")):
                result = prepare_dataset(source, scratch, question=QUESTION,
                                         settings=DatasetSettings(rows_per_shard=128), output_path=output)
            self.assertEqual(result.input_sha256, sha256(raw).hexdigest())
            self.assertEqual(result.total_rows, 259)
            self.assertEqual([(s.start_row, s.row_count) for s in result.shards],
                             [(0, 128), (128, 128), (256, 3)])
            self.assertEqual(b"".join(s.path.read_bytes() for s in result.shards), raw)
            restored = [json.loads(line) for shard in result.shards
                        for line in shard.path.read_bytes().splitlines()]
            self.assertEqual(restored, records)
            self.assertEqual(output.read_bytes(), b"previous scored output")
            self.assertEqual(decision_input(restored[0], QUESTION),
                             {"state": records[0]["content"], "question": QUESTION})
            self.assertEqual(list(json.loads(result.manifest)["question"]["criteria"]),
                             [["b", "B"], ["a", "A"]])
            self.assertEqual(result.job_id, sha256(result.manifest).hexdigest())

    def test_empty_preparation_does_not_publish(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            source, output, scratch = root / "input", root / "output", root / "scratch"
            source.touch()
            scratch.mkdir()
            result = prepare_dataset(source, scratch, question={"type": "noul"}, output_path=output)
            self.assertEqual(result.shards, ())
            self.assertEqual(result.total_rows, 0)
            self.assertEqual(result.input_sha256, sha256(b"").hexdigest())
            self.assertFalse(output.exists())
            self.assertEqual(list(scratch.iterdir()), [])

    def test_invalid_rows_report_source_position_and_clean_partial_shards(self):
        for bad in (b"\n", b"bad\n", b"[]\n", b'{}\n', b'{"content":42}\n',
                    b'{"content":"x","decision":null}\n', b'{"content":"\xff"}\n',
                    b'{"content":"x","metadata":NaN}\n',
                    b'{"content":"x","metadata":1e999}\n',
                    b'{"content":"x","content":"y"}\n'):
            with self.subTest(bad=bad), TemporaryDirectory() as temp:
                root = Path(temp)
                source, scratch = root / "input", root / "scratch"
                scratch.mkdir()
                source.write_bytes(b'{"content":"ok"}\n' * 3 + bad)
                with self.assertRaisesRegex(ValueError, "[Rr]ow 4"):
                    prepare_dataset(source, scratch, question=QUESTION,
                                    settings=DatasetSettings(rows_per_shard=2))
                self.assertEqual(list(scratch.iterdir()), [])

    def test_custom_text_field_and_original_images_are_not_rpc_images(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            source, scratch = root / "input", root / "scratch"
            scratch.mkdir()
            source.write_text('{"message":"hello","images":["original-url"],"state":42}\n')
            result = prepare_dataset(source, scratch, question=QUESTION,
                                     settings=DatasetSettings(text_field="message"))
            record = json.loads(result.shards[0].path.read_text())
            self.assertEqual(decision_input(record, QUESTION, text_field="message")["state"], "hello")
            self.assertEqual(record["state"], 42)
            self.assertEqual(record["images"], ["original-url"])

    def test_paths_and_scratch_safety(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            source, scratch = root / "input", root / "scratch"
            source.write_text('{"content":"hello"}\n')
            scratch.mkdir()
            aliases = [source, root / "." / "input", root / "symlink", root / "hardlink"]
            aliases[2].symlink_to(source)
            os.link(source, aliases[3])
            for alias in aliases:
                with self.assertRaisesRegex(ValueError, "must differ"):
                    prepare_dataset(source, scratch, question=QUESTION, output_path=alias)
            with self.assertRaisesRegex(ValueError, "outside"):
                prepare_dataset(source, scratch, question=QUESTION, output_path=scratch / "000000.jsonl")
            protected = scratch / "keep"
            protected.write_text("keep")
            with self.assertRaisesRegex(ValueError, "empty scratch"):
                prepare_dataset(source, scratch, question=QUESTION)
            self.assertEqual(protected.read_text(), "keep")
            self.assertEqual(validate_dataset_paths(source, root / "output"), (source, root / "output"))


class DatasetManifestTests(unittest.TestCase):
    def manifest(self, question=QUESTION, settings=None):
        return dataset_manifest(sha256(b"input").hexdigest(), 1, question=question,
                                settings=settings or DatasetSettings())

    def test_candidate_order_is_semantic(self):
        reordered = {**QUESTION, "criteria": {"a": "A", "b": "B"}}
        self.assertNotEqual(self.manifest(), self.manifest(question=reordered))
        # Ordinary top-level key order is not semantic.
        self.assertEqual(self.manifest(), self.manifest(question=dict(reversed(list(QUESTION.items())))))
        score = {"type": "score", "criteria": ["low", "high"]}
        self.assertNotEqual(self.manifest(question=score), self.manifest(
            question={"type": "score", "criteria": ["high", "low"]}))

    def test_semantic_settings_invalidate_identity(self):
        baseline = self.manifest()
        for field, value in (("text_field", "message"), ("batch_size", 16),
                             ("rows_per_shard", 64), ("max_padded_tokens", 32768),
                             ("processing_window", 64)):
            with self.subTest(field=field):
                self.assertNotEqual(baseline, self.manifest(settings=replace(DatasetSettings(), **{field: value})))
        for field, value in (("MODEL_ID", "different"), ("MODEL_REVISION", "different"),
                             ("PIPELINE_VERSION", 2), ("MANIFEST_VERSION", 2),
                             ("NUMERICAL_POLICY", {"weights": "different"}), ("OUTPUT_FIELD", "other")):
            with patch(f"distill_jev.models.decision.perplexity.dataset.{field}", value):
                self.assertNotEqual(baseline, self.manifest())
        self.assertNotEqual(baseline, dataset_manifest(sha256(b"other").hexdigest(), 1,
                            question=QUESTION, settings=DatasetSettings()))
        self.assertNotEqual(baseline, dataset_manifest(sha256(b"input").hexdigest(), 2,
                            question=QUESTION, settings=DatasetSettings()))
        self.assertNotEqual(baseline, self.manifest(question={**QUESTION, "instructions": "Other"}))

    def test_operational_settings_excluded_and_provenance_present(self):
        self.assertEqual(self.manifest(), self.manifest(
            settings=DatasetSettings(max_containers=8, progress_interval=0.5)))
        manifest = json.loads(self.manifest())
        self.assertNotIn("max_containers", manifest)
        self.assertNotIn("progress_interval", manifest)
        self.assertNotIn("output_path", manifest)
        self.assertEqual(manifest["processing_window"], 128)
        self.assertEqual(manifest["model_revision"], "6195aa55a72ae21e503c8a60198f7b390d42adb5")
        self.assertEqual(manifest["numerical_policy"]["probabilities"], "native-calibrated-unmodified")

    def test_invalid_settings(self):
        cases = {"batch_size": [0, 33, True, 1.5], "rows_per_shard": [0, -1, True, 1.5],
                 "max_containers": [0, True, 1.5], "processing_window": [0, 129, True],
                 "max_padded_tokens": [8191, True, 8192.5],
                 "text_field": ["", "decision", None],
                 "progress_interval": [0, -1, True, float("nan"), float("inf"), "30"]}
        for field, values in cases.items():
            for value in values:
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    DatasetSettings(**{field: value})

    def test_shared_question_validation_without_rpc_row_bound(self):
        for question in ({"type": "choice", "criteria": {"a": "A"}},
                         {"type": "score", "criteria": ["low", "high"]},
                         {"type": "noul"}, {"type": "noul", "criteria": {"true": "Yes"}}):
            validate_question(question)
            self.manifest(question=question)
        for question in (None, {}, {"type": "choice", "criteria": {}},
                         {"type": "score", "criteria": ["only"]},
                         {"type": "noul", "criteria": {"yes": "Yes"}},
                         {"type": "choice", "criteria": {1: "A"}},
                         {"type": "noul", "extra": float("nan")}):
            with self.subTest(question=question), self.assertRaises(ValueError):
                validate_question(question)
        for checksum, rows in (("bad", 1), ("a" * 64, -1), ("a" * 64, True)):
            with self.assertRaises(ValueError):
                dataset_manifest(checksum, rows, question=QUESTION, settings=DatasetSettings())
