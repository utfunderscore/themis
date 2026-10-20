import importlib.util
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING

HAS_TRAINING = all(importlib.util.find_spec(name) is not None
                   for name in ("torch", "transformers"))

if TYPE_CHECKING or HAS_TRAINING:
    import torch
    from transformers import BertTokenizer, DebertaV2Config, DebertaV2ForSequenceClassification

    from themis.models.deberta import ACTION_LABELS, LABELS
    from themis.models.deberta.inference import Moderation


@unittest.skipUnless(HAS_TRAINING, "Install the training extra to run inference tests")
class DebertaInferenceTests(unittest.TestCase):
    def test_saved_model_inference_and_independent_tasks(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            directory = root / "model"
            directory.mkdir()
            vocab = root / "vocab.txt"
            vocab.write_text("[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\nhello\nworld\n")
            BertTokenizer(vocab_file=str(vocab)).save_pretrained(directory)
            config = DebertaV2Config(
                vocab_size=7, hidden_size=16, num_hidden_layers=1,
                num_attention_heads=2, intermediate_size=32, max_position_embeddings=16,
                id2label=dict(enumerate(LABELS + ACTION_LABELS)),
            )
            model = DebertaV2ForSequenceClassification(config)
            with torch.no_grad():
                model.classifier.weight.zero_()
                model.classifier.bias.copy_(torch.tensor([0., 1, 2, 3, 4, 5, 100, 101]))
            model.save_pretrained(directory)
            (root / "run_config.json").write_text(json.dumps({"max_length": 12}))
            inference = Moderation(root, device="cpu")
            texts = ["hello", "world " * 40, ""]
            results = inference.predict(texts, batch_size=2)
            self.assertEqual(results, inference.predict(texts, batch_size=1))
            self.assertEqual(len(results), 3)
            self.assertFalse(inference.model.training)
            self.assertEqual(inference.max_length, 12)
            for result in results:
                self.assertEqual(result["category"], "threat")
                self.assertEqual(result["action"], "block_and_review")
                self.assertAlmostEqual(sum(result["category_probabilities"].values()), 1, places=6)
                self.assertAlmostEqual(sum(result["action_probabilities"].values()), 1, places=6)
                self.assertAlmostEqual(result["action_probabilities"]["allow"], 0.2689414)
            self.assertEqual(inference.predict([]), [])
            with self.assertRaises(TypeError):
                inference.predict("hello")  # pyright: ignore[reportArgumentType]
            with self.assertRaises(ValueError):
                inference.predict(texts, batch_size=0)
            with self.assertRaises(ValueError):
                Moderation(directory, max_length=17)
            (root / "run_config.json").unlink()
            with self.assertRaisesRegex(ValueError, "Supply max_length"):
                Moderation(directory)
            self.assertEqual(Moderation(directory, device="cpu", max_length=12).predict([]), [])
            inference.tokenizer.save_pretrained(directory)
            self.assertEqual(Moderation(directory, device="cpu").max_length, 12)
