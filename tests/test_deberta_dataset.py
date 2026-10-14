import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import polars as pl

from distill_jev.models.deberta import LABELS, prepare_dataset


class DebertaDatasetTests(unittest.TestCase):
    def test_three_files_targets_and_grouped_splits(self):
        rows = []
        for label in range(6):
            probabilities = [0.02] * 6
            probabilities[label] = 0.9
            for index in range(20):
                rows.append({"content": f"Message {label} #{index} 🌍!?",
                             "decision": {"answer": {"probabilities": dict(reversed(list(zip(LABELS, probabilities))))}}})
        rows[1]["content"] = "MESSAGE 0 #0 🌍!?  "
        rows.append({**rows[0], "content": "   "})
        with TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "input.jsonl"
            source.write_text("".join(json.dumps(row) + "\n" for row in rows))
            output = root / "splits"
            split_paths = prepare_dataset(source, output)
            self.assertEqual(split_paths, {split: output / f"{split}.parquet"
                                          for split in ("train", "validation", "test")})
            self.assertEqual({p.name for p in output.iterdir()}, {"train.parquet", "validation.parquet", "test.parquet"})
            frames = [pl.read_parquet(path) for path in split_paths.values()]
            self.assertEqual(sum(frame.height for frame in frames), 120)
            combined = pl.concat(frames).sort("content")
            self.assertEqual(combined["content"].to_list(), sorted(row["content"] for row in rows[:-1]))
            groups = [{" ".join(text.casefold().split()) for text in frame["content"]} for frame in frames]
            for a, b in ((0, 1), (0, 2), (1, 2)):
                self.assertFalse(groups[a] & groups[b])
            for frame in frames:
                self.assertEqual(frame.columns, ["content", "targets", "teacher_argmax"])
                self.assertEqual(frame.schema["targets"], pl.Array(pl.Float32, 6))
                self.assertEqual(set(frame["teacher_argmax"]), set(range(6)))
                for row in frame.iter_rows(named=True):
                    self.assertEqual(row["targets"].index(max(row["targets"])), row["teacher_argmax"])
            second = root / "second"
            prepare_dataset(source, second)
            for path in output.iterdir():
                self.assertEqual(path.read_bytes(), (second / path.name).read_bytes())
            with self.assertRaises(FileExistsError):
                prepare_dataset(source, output)
