import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import polars as pl

from themis.models.deberta import ACTION_LABELS, LABELS
from themis.models.deberta.correction import obfuscation_key, prepare_correction_dataset


def _write_teacher_splits(directory: Path, size: int) -> None:
    directory.mkdir()
    for split in ("train", "validation", "test"):
        pl.DataFrame({
            "content": [f"teacher {split} {index}" for index in range(size)],
            "targets": [[0.7, 0.1, 0.05, 0.05, 0.05, 0.05]] * size,
            "teacher_argmax": [0] * size,
            "action_targets": [[0.9, 0.1]] * size,
            "action_teacher_argmax": [0] * size,
        }).with_columns(
            pl.col("targets").cast(pl.Array(pl.Float32, 6)),
            pl.col("teacher_argmax").cast(pl.UInt8),
            pl.col("action_targets").cast(pl.Array(pl.Float32, 2)),
            pl.col("action_teacher_argmax").cast(pl.UInt8),
        ).write_parquet(directory / f"{split}.parquet")


class CorrectionDatasetTests(unittest.TestCase):
    def test_obfuscated_variants_share_a_key(self):
        self.assertEqual(obfuscation_key("Go h0me n0w!"), obfuscation_key("go H.O.M.E now"))
        self.assertEqual(obfuscation_key("p@$$w0rd"), "password")
        self.assertNotEqual(obfuscation_key("hello there"), obfuscation_key("hello world"))

    def test_splits_targets_replay_and_grouping(self):
        rows = []
        for category in ("none", "slur", "sexual"):
            for index in range(40):
                rows.append({"text": f"{category} message number {index}", "category": category})
                rows.append({"text": f"{category.upper()} m3ssage NUMBER {index}!", "category": category})
        rows.append(dict(rows[0]))  # Exact duplicate.
        with TemporaryDirectory() as temp:
            root = Path(temp)
            labeled = root / "labeled.jsonl"
            labeled.write_text("".join(json.dumps(row) + "\n" for row in rows))
            teacher = root / "teacher"
            _write_teacher_splits(teacher, 1000)
            paths = prepare_correction_dataset(labeled, teacher, root / "out", replay_ratio=2.0,
                                               label_smoothing=0.06)
            self.assertEqual(set(paths), {"train", "validation", "test", "teacher_test"})
            frames = {split: pl.read_parquet(path) for split, path in paths.items()}
            reviewed = {split: frame.filter(~pl.col("content").str.starts_with("teacher"))
                        for split, frame in frames.items()}
            self.assertEqual(sum(frame.height for frame in reviewed.values()), 240)
            self.assertEqual(frames["teacher_test"].height, 1000)
            self.assertEqual(reviewed["teacher_test"].height, 0)
            self.assertEqual(frames["test"].height, reviewed["test"].height)
            for split in ("train", "validation"):
                self.assertEqual(frames[split].height - reviewed[split].height, 2 * reviewed[split].height)
            keys = {split: {obfuscation_key(text) for text in frame["content"]}
                    for split, frame in reviewed.items() if split != "teacher_test"}
            self.assertFalse(keys["train"] & keys["test"])
            self.assertFalse(keys["train"] & keys["validation"])
            self.assertFalse(keys["validation"] & keys["test"])
            slur = reviewed["train"].filter(pl.col("content").str.starts_with("slur")).row(0, named=True)
            expected = [0.01] * 6
            expected[LABELS.index("slur")] = 0.95
            np.testing.assert_allclose(slur["targets"], expected, rtol=1e-6)
            self.assertEqual(slur["teacher_argmax"], LABELS.index("slur"))
            np.testing.assert_allclose(slur["action_targets"], [0.03, 0.97], rtol=1e-6)
            self.assertEqual(slur["action_teacher_argmax"], ACTION_LABELS.index("block_and_review"))
            manifest = json.loads((root / "out" / "manifest.json").read_text())
            self.assertEqual(manifest["counts"]["teacher_test"], {"reviewed": 0, "teacher": 1000})

    def test_rejects_conflicts_and_categories_without_actions(self):
        cases = (
            ([{"text": "a", "category": "slur"}, {"text": "a", "category": "none"}], "Conflicting"),
            ([{"text": "a", "category": "insult"}], "action"),
            ([{"text": "a", "category": "unknown"}], "Unknown category"),
        )
        with TemporaryDirectory() as temp:
            root = Path(temp)
            teacher = root / "teacher"
            _write_teacher_splits(teacher, 10)
            labeled = root / "labeled.jsonl"
            for rows, message in cases:
                labeled.write_text("".join(json.dumps(row) + "\n" for row in rows))
                with self.assertRaisesRegex(ValueError, message):
                    prepare_correction_dataset(labeled, teacher, root / "out")
            labeled.write_text(json.dumps({"text": "a", "category": "insult", "action": "allow"}) + "\n")
            prepare_correction_dataset(labeled, teacher, root / "out")


class CorrectStudentWorkflowTests(unittest.TestCase):
    def test_reuses_run_only_for_the_same_starting_model(self):
        from unittest.mock import patch

        from themis.models.deberta.workflow import CORRECTION_TRAINING, correct_student

        with TemporaryDirectory() as temp:
            root = Path(temp)
            splits = root / "correction"
            first, second = root / "first", root / "second"
            with (patch("themis.models.deberta.correction.prepare_correction_dataset",
                        side_effect=lambda *args: (splits.mkdir(), (splits / "manifest.json").touch())) as prepare,
                  patch("themis.models.deberta.modal_training.train",
                        side_effect=[{"run_id": "one"}, {"run_id": "two"}]) as training,
                  patch("themis.models.deberta.workflow.download_student") as download):
                correct_student("labels.jsonl", first, correction_data_directory=splits)
                training.assert_called_once_with(split_directory=splits, config=CORRECTION_TRAINING,
                                                 init_model_directory=first)
                correct_student("labels.jsonl", first, correction_data_directory=splits)
                self.assertEqual(training.call_count, 1)
                correct_student("labels.jsonl", second, correction_data_directory=splits)
                self.assertEqual(training.call_count, 2)
                prepare.assert_called_once()
                self.assertEqual(download.call_count, 3)
            receipt = json.loads((splits / "modal-run.json").read_text())
            self.assertEqual(receipt, {"run_id": "two", "init_model_directory": str(second.resolve())})
            self.assertTrue((splits / "modal-run-one.json").is_file())

    def test_correct_and_export_exports_the_corrected_model(self):
        from unittest.mock import patch

        from themis.models.deberta.workflow import correct_and_export

        with TemporaryDirectory() as temp:
            root = Path(temp)
            distilled, corrected = root / "a" / "model", root / "b" / "model"
            with (patch("themis.models.deberta.workflow.correct_student", return_value=corrected) as correct,
                  patch("themis.models.deberta.workflow.export_student") as export):
                correct_and_export("labels.jsonl", distilled, teacher_split_directory="splits",
                                   correction_data_directory="fixes")
            correct.assert_called_once_with("labels.jsonl", distilled, teacher_split_directory="splits",
                                            correction_data_directory="fixes")
            export.assert_called_once_with(model_directory=corrected, output_directory=root / "b" / "onnx")
