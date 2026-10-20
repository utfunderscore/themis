import importlib.util
import json
import unittest
from contextlib import chdir
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, cast
from unittest.mock import patch

import polars as pl

HAS_TRAINING = all(importlib.util.find_spec(name) is not None
                   for name in ("torch", "transformers", "accelerate"))

if TYPE_CHECKING or HAS_TRAINING:
    import numpy as np
    import torch
    from transformers import BertTokenizerFast, DebertaV2Config, DebertaV2ForSequenceClassification

    from themis.models.deberta import ACTION_LABELS, LABELS
    from themis.models.deberta.training import (
        SoftTargetDataset,
        TrainingConfig,
        soft_target_loss,
        multitask_loss,
        train,
    )


@unittest.skipUnless(HAS_TRAINING, "Install the training extra to run trainer tests")
class DebertaTrainingTests(unittest.TestCase):
    def test_pipeline_training_prepares_reuses_and_preserves_outputs(self):
        from themis.models.deberta import BASE_MODEL
        from themis.models.deberta.workflow import train_student

        run = {"run_id": "test", "model_name": BASE_MODEL}
        with TemporaryDirectory() as temp, chdir(temp):
            source = Path("reviews.jsonl")
            splits = Path("cache/deberta/review-splits")
            receipt = Path("cache/deberta/review-modal-run.json")
            with (patch("themis.models.deberta.workflow.prepare_training_dataset") as prepare,
                  patch("themis.models.deberta.workflow.download_student"),
                  patch("themis.models.deberta.modal_training.train", return_value=run) as training):
                splits.parent.mkdir(parents=True)
                train_student(source)
                prepare.assert_called_once_with(source_path=source, output_directory=splits)
                training.assert_called_once_with(split_directory=splits)
                self.assertEqual(json.loads(receipt.read_text()), run)
                receipt.unlink()
                splits.mkdir()
                for split in ("train", "validation", "test"):
                    (splits / f"{split}.parquet").touch()
                prepare.reset_mock()
                train_student(source)
                prepare.assert_not_called()
                self.assertEqual(training.call_count, 2)
                training.reset_mock()
                train_student(source)
                training.assert_not_called()

    def test_pipeline_retrains_runs_from_another_base_model(self):
        from themis.models.deberta import BASE_MODEL
        from themis.models.deberta.workflow import train_student

        with TemporaryDirectory() as temp, chdir(temp):
            splits = Path("cache/deberta/review-splits")
            splits.mkdir(parents=True)
            for split in ("train", "validation", "test"):
                (splits / f"{split}.parquet").touch()
            receipt = Path("cache/deberta/review-modal-run.json")
            # Receipts without model_name predate mDeBERTa and are DeBERTa-v3 runs.
            legacy = {"run_id": "old"}
            receipt.write_text(json.dumps(legacy))
            run = {"run_id": "new", "model_name": BASE_MODEL}
            with (patch("themis.models.deberta.workflow.download_student") as download,
                  patch("themis.models.deberta.modal_training.train", return_value=run) as training):
                train_student("reviews.jsonl")
                training.assert_called_once_with(split_directory=splits)
                download.assert_called_once_with(result=run)
            self.assertEqual(json.loads(receipt.read_text()), run)
            self.assertEqual(json.loads(Path("cache/deberta/review-modal-run-old.json").read_text()), legacy)

    def test_tasks_have_independent_normalization_and_gradients(self):
        logits = torch.zeros((1, 8), requires_grad=True)
        targets = torch.tensor([[1., 0, 0, 0, 0, 0, 0.25, 0.75]])
        loss = multitask_loss(logits, targets)
        self.assertAlmostEqual(loss.item(), np.log(6) + np.log(2), places=6)
        loss.backward()
        assert logits.grad is not None
        torch.testing.assert_close(logits.grad[:, :6], torch.full((1, 6), 1 / 6) - targets[:, :6])
        torch.testing.assert_close(logits.grad[:, 6:], torch.tensor([[0.25, -0.25]]))

    def test_soft_targets_preserve_distribution_and_gradients(self):
        logits = torch.zeros((2, 6), requires_grad=True)
        targets = torch.tensor([[0.7, 0.3, 0, 0, 0, 0], [0, 0, 0, 0, 0.2, 0.8]])
        loss = soft_target_loss(logits, targets)
        self.assertAlmostEqual(loss.item(), np.log(6), places=6)
        loss.backward()
        torch.testing.assert_close(logits.grad, (torch.full_like(targets, 1 / 6) - targets) / 2)

    def test_reject_invalid_probabilities_before_tokenizing(self):
        with TemporaryDirectory() as temp:
            path = Path(temp) / "bad.parquet"
            for targets in ([0.5] * 6, [float("nan")] * 6, [-1, 2, 0, 0, 0, 0]):
                pl.DataFrame({"content": ["hello"], "targets": [targets],
                              "teacher_argmax": [0]}).write_parquet(path)
                with self.assertRaisesRegex(ValueError, "probability distributions"):
                    SoftTargetDataset(path, None, 16)

    def test_train_select_evaluate_and_reload_offline(self):
        # Real DeBERTa forward/backward and Trainer checkpoints, using tiny local
        # weights and a local tokenizer to avoid Hub downloads or GPU requirements.
        with TemporaryDirectory() as temp:
            root = Path(temp)
            base = root / "base"
            base.mkdir()
            vocabulary = base / "vocab.txt"
            vocabulary.write_text("[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\nhello\nworld\n")
            tokenizer = BertTokenizerFast(vocab_file=str(vocabulary))
            tokenizer.save_pretrained(base)
            model = DebertaV2ForSequenceClassification(DebertaV2Config(
                vocab_size=7, hidden_size=16, num_hidden_layers=1, num_attention_heads=2,
                intermediate_size=24, max_position_embeddings=32, num_labels=8,
                relative_attention=False, pad_token_id=0,
            ))
            model.save_pretrained(base)
            splits = root / "splits"
            splits.mkdir()
            for split in ("train", "validation", "test", "teacher_test"):
                pl.DataFrame({
                    "content": ["hello", "hello world", "world", "world hello"],
                    "targets": [[0.7, 0.1, 0.05, 0.05, 0.05, 0.05]] * 4,
                    "teacher_argmax": [0] * 4,
                    "action_targets": [[0.25, 0.75]] * 4,
                    "action_teacher_argmax": [1] * 4,
                }).write_parquet(splits / f"{split}.parquet")
            output = root / "run"
            result = train(splits, output, config=TrainingConfig(
                model_name=str(base), max_length=16, epochs=2,
                train_batch_size=1, eval_batch_size=2,
                gradient_accumulation_steps=2, use_cpu=True,
            ))
            metrics = json.loads((output / "metrics.json").read_text())
            self.assertTrue(np.isfinite(metrics["test"]["test_loss"]))
            self.assertIn("validation_category_teacher_kl", metrics["validation"])
            self.assertIn("validation_action_teacher_kl", metrics["validation"])
            self.assertIn("teacher_test_category_teacher_agreement", metrics["teacher_test"])
            self.assertTrue(Path(metrics["best_checkpoint"]).is_dir())
            model_directory = cast(Path, result["model_directory"])
            tokenizer_config = json.loads((model_directory / "tokenizer_config.json").read_text())
            self.assertEqual(tokenizer_config["model_max_length"], 16)
            saved = DebertaV2ForSequenceClassification.from_pretrained(model_directory)
            best = DebertaV2ForSequenceClassification.from_pretrained(metrics["best_checkpoint"])
            assert saved.config.id2label is not None
            self.assertEqual(tuple(saved.config.id2label.values()), LABELS + ACTION_LABELS)
            for key, value in saved.state_dict().items():
                torch.testing.assert_close(value, best.state_dict()[key])
            self.assertTrue(any(not torch.equal(value, model.state_dict()[key])
                                for key, value in saved.state_dict().items()))
            self.assertTrue((output / "model" / "tokenizer_config.json").is_file())
            with self.assertRaises(FileExistsError):
                train(splits, output)

    def test_local_training_then_post_training_and_reuse(self):
        from themis.models.deberta import workflow

        with TemporaryDirectory() as temp:
            root = Path(temp)
            base = root / "base"
            base.mkdir()
            vocabulary = base / "vocab.txt"
            vocabulary.write_text("[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\nhello\nworld\n")
            BertTokenizerFast(vocab_file=str(vocabulary)).save_pretrained(base)
            DebertaV2ForSequenceClassification(DebertaV2Config(
                vocab_size=7, hidden_size=16, num_hidden_layers=1, num_attention_heads=2,
                intermediate_size=24, max_position_embeddings=32, num_labels=8,
                relative_attention=False, pad_token_id=0,
            )).save_pretrained(base)
            splits = root / "splits"
            splits.mkdir()
            for split in ("train", "validation", "test"):
                pl.DataFrame({
                    "content": ["hello", "hello world", "world", "world hello"],
                    "targets": [[0.7, 0.1, 0.05, 0.05, 0.05, 0.05]] * 4,
                    "teacher_argmax": [0] * 4,
                    "action_targets": [[0.75, 0.25]] * 4,
                    "action_teacher_argmax": [0] * 4,
                }).with_columns(
                    pl.col("targets").cast(pl.Array(pl.Float32, 6)),
                    pl.col("teacher_argmax").cast(pl.UInt8),
                    pl.col("action_targets").cast(pl.Array(pl.Float32, 2)),
                    pl.col("action_teacher_argmax").cast(pl.UInt8),
                ).write_parquet(splits / f"{split}.parquet")
            labeled = root / "labeled.jsonl"
            labeled.write_text("".join(
                json.dumps({"text": f"{word} " * count, "category": category}) + "\n"
                for word, category in (("hello", "none"), ("world", "slur")) for count in range(1, 11)
            ))
            config = {"max_length": 16, "epochs": 1, "train_batch_size": 2, "use_cpu": True}
            with patch.object(workflow, "BASE_MODEL", str(base)):
                model = workflow.train_student_locally(
                    "unused.jsonl", training_data_directory=splits,
                    output_directory=root / "run", config=config,
                )
                self.assertEqual(model, root / "run" / "model")
                with patch("themis.models.deberta.training.train") as training:
                    self.assertEqual(workflow.train_student_locally(
                        "unused.jsonl", training_data_directory=splits, output_directory=root / "run",
                    ), model)
                    training.assert_not_called()
            with self.assertRaises(FileExistsError):
                workflow.train_student_locally("unused.jsonl", training_data_directory=splits,
                                               output_directory=root / "run")
            corrected = workflow.correct_student_locally(
                labeled, model, teacher_split_directory=splits,
                correction_data_directory=root / "correction", output_directory=root / "fixed", config=config,
            )
            run_config = json.loads((root / "fixed" / "run_config.json").read_text())
            self.assertEqual(run_config["model_name"], str(model.resolve()))
            self.assertEqual(run_config["learning_rate"], workflow.CORRECTION_TRAINING["learning_rate"])
            self.assertIn("teacher_test", json.loads((root / "fixed" / "metrics.json").read_text()))
            self.assertEqual(workflow.correct_student_locally(
                labeled, model, teacher_split_directory=splits,
                correction_data_directory=root / "correction", output_directory=root / "fixed",
            ), corrected)
            with self.assertRaises(FileExistsError):
                workflow.correct_student_locally(labeled, base, teacher_split_directory=splits,
                                                 correction_data_directory=root / "correction",
                                                 output_directory=root / "fixed")
            with patch("builtins.print") as output:
                comparison = workflow.compare_correction_locally(
                    model, corrected, correction_data_directory=root / "correction", config=config,
                )
            fixed = json.loads((root / "fixed" / "metrics.json").read_text())
            self.assertEqual(comparison["post_trained"]["test"], fixed["test"])
            for split in ("test", "teacher_test"):
                self.assertTrue(np.isfinite(comparison["distilled"][split][f"{split}_loss"]))
            self.assertIn("teacher_test_category_teacher_agreement", comparison["distilled"]["teacher_test"])
            self.assertIn("distilled", output.call_args_list[-1].args[0])
            with patch("themis.models.deberta.training.evaluate") as evaluation:
                self.assertEqual(workflow.compare_correction_locally(
                    model, corrected, correction_data_directory=root / "correction",
                ), comparison)
                evaluation.assert_not_called()
