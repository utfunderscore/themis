import importlib.util
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory


HAS_EXPORT = all(importlib.util.find_spec(name) is not None
                 for name in ("torch", "transformers", "onnx", "onnxruntime"))


@unittest.skipUnless(HAS_EXPORT, "Install training and onnx extras")
class OnnxExportTests(unittest.TestCase):
    def test_relative_attention_export_and_bundle(self):
        from transformers import BertTokenizer, DebertaV2Config, DebertaV2ForSequenceClassification

        from themis.models.deberta.dataset import ACTION_LABELS, LABELS
        from themis.models.deberta.onnx_export import export_onnx

        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "model"
            source.mkdir()
            vocab = root / "vocab.txt"
            vocab.write_text("[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\nhello\nworld\n")
            BertTokenizer(vocab_file=str(vocab)).save_pretrained(source)
            config = DebertaV2Config(
                vocab_size=7, hidden_size=16, num_hidden_layers=1,
                num_attention_heads=2, intermediate_size=32, max_position_embeddings=32,
                relative_attention=True, pos_att_type=["p2c", "c2p"],
                position_buckets=8, max_relative_positions=32,
                id2label=dict(enumerate(LABELS + ACTION_LABELS)),
            )
            DebertaV2ForSequenceClassification(config).save_pretrained(source)
            (root / "run_config.json").write_text(json.dumps({"max_length": 24}))
            destination = root / "onnx"
            path = export_onnx(root, destination)
            self.assertTrue(path.is_file())
            metadata = json.loads((destination / "inference_config.json").read_text())
            self.assertEqual(metadata["max_length"], 24)
            self.assertEqual(metadata["output_groups"]["action"]["start"], len(LABELS))
            tokenizer_config = json.loads((destination / "tokenizer_config.json").read_text())
            self.assertEqual(tokenizer_config["model_max_length"], 24)
            fixtures = json.loads((destination / "validation_examples.json").read_text())
            self.assertEqual(len(fixtures[-1]["input_ids"][0]), 24)
            with self.assertRaises(FileExistsError):
                export_onnx(root, destination)
            with self.assertRaises(ValueError):
                export_onnx(root, root / "invalid", max_length=100)
            self.assertFalse((root / "invalid").exists())
