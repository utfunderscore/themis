import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from themis.models.decision.perplexity.dataset import split_dataset
from themis.datasets.review import generate_review_dataset


class ReviewDatasetTests(unittest.TestCase):
    def test_native_rows_preserve_exact_context_and_native_fields(self):
        rows = [
            {
                "state": "native chat",
                "content": "not the native input",
                "question": {"type": "choice", "criteria": {"insult": "Insult"}},
                "images": ["data:image/png;base64,abc"],
                "source": {"content": "original chat", "metadata": {"only_first": 1}},
                "decision": {"answer": {"probabilities": {"insult": 0.5, "none": 0.5}}},
            },
            {
                "state": "second chat ☃",
                "content": "second content",
                "question": {"type": "choice", "criteria": {"insult": "Insult"}},
                "images": [],
                "source": {"content": "second original", "metadata": {"only_first": 2}},
                "decision": {"answer": {"probabilities": {"insult": 0.3, "none": 0.7}}},
            },
        ]
        with TemporaryDirectory() as directory:
            source = Path(directory) / "scores.jsonl"
            destination = Path(directory) / "nested" / "review.jsonl"
            source.write_text("".join(json.dumps(row) + "\n" for row in rows))
            original_bytes = source.read_bytes()
            generate_review_dataset(source, destination)
            prepared = [json.loads(line) for line in destination.read_text().splitlines()]
            self.assertEqual([row["source"] for row in prepared], rows)
            self.assertEqual([row["state"] for row in prepared], ["native chat", "second chat ☃"])
            self.assertEqual([row["images"] for row in prepared], [row["images"] for row in rows])
            for row in prepared:
                self.assertEqual(set(row), {"images", "state", "question", "source"})
            self.assertIn("classified as insult.", prepared[0]["question"]["instructions"])
            self.assertIn("classified as none.", prepared[1]["question"]["instructions"])
            self.assertEqual(source.read_bytes(), original_bytes)
            first_output = destination.read_bytes()
            generate_review_dataset(source, destination)
            self.assertEqual(destination.read_bytes(), first_output)
            with TemporaryDirectory(dir=directory) as shards_directory:
                shards, _, total = split_dataset(destination, shards_directory, rows_per_shard=1)
                self.assertEqual(total, 2)
                self.assertEqual(len(shards), 2)

    def test_legacy_rows_use_content_as_state(self):
        rows = [
            {
                "content": "legacy chat ☃",
                "metadata": {"kept": "preserved"},
                "decision": {"answer": {"probabilities": {"none": 0.7, "threat": 0.3}}},
            },
        ]
        with TemporaryDirectory() as directory:
            source, destination = Path(directory) / "scores.jsonl", Path(directory) / "review.jsonl"
            source.write_text("".join(json.dumps(row) + "\n" for row in rows))
            generate_review_dataset(source, destination)
            [prepared] = [json.loads(line) for line in destination.read_text().splitlines()]
            self.assertEqual(prepared["source"], rows[0])
            self.assertEqual(prepared["state"], "legacy chat ☃")
            self.assertEqual(prepared["metadata"], rows[0]["metadata"])
            self.assertNotIn("content", prepared)
            self.assertIn("classified as none.", prepared["question"]["instructions"])

    def test_empty_input_publishes_empty_output(self):
        with TemporaryDirectory() as directory:
            source, destination = Path(directory) / "scores.jsonl", Path(directory) / "review.jsonl"
            source.touch()
            destination.write_text("old output")
            self.assertEqual(generate_review_dataset(source, destination), destination)
            self.assertEqual(destination.read_bytes(), b"")

    def test_invalid_rows_do_not_replace_existing_output(self):
        valid = {"state": "chat", "decision": {"answer": {"probabilities": {"none": 1.0}}}}
        invalid_rows = [
            "not json", "null", "[]", "{}",
            json.dumps({"state": "chat"}),
            json.dumps({"state": "chat", "decision": {"answer": None}}),
            json.dumps({**valid, "state": None}),
            json.dumps({"content": 123, "decision": valid["decision"]}),
            json.dumps({**valid, "metadata": float("nan")}),
        ]
        for probabilities in ({}, [], {"": 1}, {"none": None}, {"none": True},
                              {"none": "1"}, {"none": -0.1}, {"none": 1.1},
                              {"none": float("nan")}, {"none": float("inf")}):
            invalid_rows.append(json.dumps({
                "state": "chat", "decision": {"answer": {"probabilities": probabilities}},
            }))
        with TemporaryDirectory() as directory:
            source, destination = Path(directory) / "scores.jsonl", Path(directory) / "review.jsonl"
            for invalid in invalid_rows:
                with self.subTest(row=invalid):
                    source.write_text(json.dumps(valid) + "\n" + invalid + "\n")
                    destination.write_text("completed output")
                    with self.assertRaises(ValueError):
                        generate_review_dataset(source, destination)
                    self.assertEqual(destination.read_text(), "completed output")
                    self.assertEqual(sorted(path.name for path in Path(directory).iterdir()),
                                     ["review.jsonl", "scores.jsonl"])

    def test_out_of_range_probabilities_report_row(self):
        with TemporaryDirectory() as directory:
            source, destination = Path(directory) / "scores.jsonl", Path(directory) / "review.jsonl"
            source.write_text("".join(json.dumps({
                "state": "chat", "decision": {"answer": {"probabilities": {"none": score}}},
            }) + "\n" for score in (1.0, 1.1)))
            with self.assertRaisesRegex(ValueError, "Row 2"):
                generate_review_dataset(source, destination)

    def test_source_aliases_are_rejected(self):
        with TemporaryDirectory() as directory:
            source = Path(directory) / "scores.jsonl"
            source.write_text("original")
            symlink, hardlink = Path(directory) / "symlink.jsonl", Path(directory) / "hardlink.jsonl"
            symlink.symlink_to(source)
            hardlink.hardlink_to(source)
            for destination in (source, symlink, hardlink):
                with self.subTest(destination=destination):
                    with self.assertRaisesRegex(ValueError, "source and destination must differ"):
                        generate_review_dataset(source, destination)
                    self.assertEqual(source.read_text(), "original")
