import importlib.util
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import polars as pl

HAS_TRAINING = all(importlib.util.find_spec(name) is not None
                   for name in ("torch", "transformers", "accelerate"))

if HAS_TRAINING:
    import numpy as np
    import torch
    from transformers import BertTokenizerFast, DebertaV2Config, DebertaV2ForSequenceClassification

    from distill_jev.models.deberta import LABELS
    from distill_jev.models.deberta.training import (
        SoftTargetDataset,
        TrainingConfig,
        soft_target_loss,
        train,
    )


@unittest.skipUnless(HAS_TRAINING, "Install the training extra to run trainer tests")
class DebertaTrainingTests(unittest.TestCase):
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
                intermediate_size=24, max_position_embeddings=32, num_labels=6,
                relative_attention=False, pad_token_id=0,
            ))
            model.save_pretrained(base)
            splits = root / "splits"
            splits.mkdir()
            for split in ("train", "validation", "test"):
                pl.DataFrame({
                    "content": ["hello", "hello world", "world", "world hello"],
                    "targets": [[0.7, 0.1, 0.05, 0.05, 0.05, 0.05]] * 4,
                    "teacher_argmax": [0] * 4,
                }).write_parquet(splits / f"{split}.parquet")
            output = root / "run"
            result = train(splits, output, config=TrainingConfig(
                model_name=str(base), max_length=16, epochs=2,
                train_batch_size=1, eval_batch_size=2,
                gradient_accumulation_steps=2, use_cpu=True,
            ))
            metrics = json.loads((output / "metrics.json").read_text())
            self.assertTrue(np.isfinite(metrics["test"]["test_loss"]))
            self.assertIn("validation_teacher_kl", metrics["validation"])
            self.assertTrue(Path(metrics["best_checkpoint"]).is_dir())
            saved = DebertaV2ForSequenceClassification.from_pretrained(result["model_directory"])
            best = DebertaV2ForSequenceClassification.from_pretrained(metrics["best_checkpoint"])
            self.assertEqual(tuple(saved.config.id2label.values()), LABELS)
            for key, value in saved.state_dict().items():
                torch.testing.assert_close(value, best.state_dict()[key])
            self.assertTrue(any(not torch.equal(value, model.state_dict()[key])
                                for key, value in saved.state_dict().items()))
            self.assertTrue((output / "model" / "tokenizer_config.json").is_file())
            with self.assertRaises(FileExistsError):
                train(splits, output)
