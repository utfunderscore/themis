import json
import math
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from distill_jev.datasets.sampling import guided_sample_dataset


class GuidedSamplingTests(unittest.TestCase):
    def make_rows(self):
        rows = []
        for label, other in (("A", "B"), ("B", "A")):
            for index in range(30):
                probability = 0.51 + index * 0.01
                rows.append({
                    "id": f"{label}{index}", "content": label + "🌍" * index,
                    "moderation": {label: probability, other: 1 - probability},
                    "other": {"original": index},
                })
        return rows

    def write_rows(self, path, rows):
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")

    def load_rows(self, path):
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

    def test_exact_quotas_disjoint_tiers_preserved_fields_and_order(self):
        with TemporaryDirectory() as directory:
            source, output = Path(directory) / "source.jsonl", Path(directory) / "sample.jsonl"
            rows = self.make_rows()
            self.write_rows(source, rows)
            original = source.read_bytes()
            self.assertEqual(guided_sample_dataset(source, output, {"A": 20, "B": 20}), output)
            selected = self.load_rows(output)
            indices = [row["sampling"]["source_row_index"] for row in selected]
            self.assertEqual(indices, sorted(set(indices)))
            self.assertEqual(len(selected), 40)
            for row in selected:
                sampling = row.pop("sampling")
                self.assertEqual(row, rows[sampling["source_row_index"]])
                self.assertEqual(sampling["content_characters"], len(row["content"]))
            selected = self.load_rows(output)
            for label in ("A", "B"):
                category = [row for row in selected if row["sampling"]["category"] == label]
                self.assertEqual(len(category), 20)
                methods = {method: [row["id"] for row in category if row["sampling"]["method"] == method]
                           for method in ("ambiguous", "long", "random")}
                self.assertEqual(methods["ambiguous"], [f"{label}{index}" for index in range(4)])
                self.assertEqual(methods["long"], [f"{label}28", f"{label}29"])
                self.assertEqual(len(methods["random"]), 14)
            manifest = json.loads(Path(str(output) + ".manifest.json").read_text())
            self.assertEqual(manifest["selected_tiers"], {
                "A": {"ambiguous": 4, "long": 2, "random": 14},
                "B": {"ambiguous": 4, "long": 2, "random": 14},
            })
            self.assertEqual(manifest["source_rows"], 60)
            self.assertEqual(manifest["sample_rows"], 40)
            self.assertEqual(manifest["available_categories"], {"A": 30, "B": 30})
            self.assertEqual(source.read_bytes(), original)
            self.assertFalse(list(Path(directory).glob("guided-sample-*")))

    def test_seed_and_quota_order_are_reproducible(self):
        with TemporaryDirectory() as directory:
            source = Path(directory) / "source.jsonl"
            self.write_rows(source, self.make_rows())
            first, second, third = [Path(directory) / f"{index}.jsonl" for index in range(3)]
            guided_sample_dataset(source, first, {"A": 20, "B": 20}, seed=42)
            guided_sample_dataset(source, second, {"B": 20, "A": 20}, seed=42)
            guided_sample_dataset(source, third, {"A": 20, "B": 20}, seed=43)
            self.assertEqual(first.read_bytes(), second.read_bytes())
            self.assertNotEqual(first.read_bytes(), third.read_bytes())

    def test_sparse_categories_rounding_ties_and_zero_quota(self):
        rows = [
            {"content": "same length", "moderation": {"A": 0.5, "B": 0.5}},
            {"content": "same length", "moderation": {"B": 0.5, "A": 0.5}},
            {"content": "same length", "moderation": {"A": 0.5, "B": 0.5}},
            {"content": "single", "moderation": {"C": 1.0}},
        ]
        with TemporaryDirectory() as directory:
            source, output = Path(directory) / "source.jsonl", Path(directory) / "sample.jsonl"
            self.write_rows(source, rows)
            guided_sample_dataset(source, output, {"A": 1, "B": 1, "C": 1, "absent": 0},
                                  ambiguous_fraction=1.0, long_fraction=0.0)
            selected = self.load_rows(output)
            self.assertEqual([row["sampling"]["source_row_index"] for row in selected], [0, 1, 3])
            self.assertTrue(all(row["sampling"]["method"] == "ambiguous" for row in selected))
            all_output = Path(directory) / "all.jsonl"
            guided_sample_dataset(source, all_output, {"A": 2, "B": 1, "C": 1})
            self.assertEqual(len(self.load_rows(all_output)), 4)
            self.assertTrue(all(row["sampling"]["method"] == "random" for row in self.load_rows(all_output)))

    def test_reuse_validates_hashes_and_does_not_rewrite(self):
        with TemporaryDirectory() as directory:
            source, output = Path(directory) / "source.jsonl", Path(directory) / "sample.jsonl"
            self.write_rows(source, self.make_rows())
            guided_sample_dataset(source, output, {"A": 20})
            original, modified = output.read_bytes(), output.stat().st_mtime_ns
            self.assertEqual(guided_sample_dataset(source, output, {"A": 20}), output)
            self.assertEqual(output.stat().st_mtime_ns, modified)
            with self.assertRaises(FileExistsError):
                guided_sample_dataset(source, output, {"A": 19})
            source.write_bytes(source.read_bytes() + b"\n")
            with self.assertRaisesRegex(ValueError, "source changed"):
                guided_sample_dataset(source, output, {"A": 20})
            self.assertEqual(output.read_bytes(), original)

    def test_output_corruption_and_unmanaged_files_are_not_overwritten(self):
        with TemporaryDirectory() as directory:
            source, output = Path(directory) / "source.jsonl", Path(directory) / "sample.jsonl"
            self.write_rows(source, self.make_rows())
            output.write_text("user work")
            with self.assertRaises(FileExistsError):
                guided_sample_dataset(source, output, {"A": 20})
            self.assertEqual(output.read_text(), "user work")
            managed = Path(directory) / "managed.jsonl"
            guided_sample_dataset(source, managed, {"A": 20})
            managed.write_text("corrupt")
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                guided_sample_dataset(source, managed, {"A": 20})
            self.assertEqual(managed.read_text(), "corrupt")

    def test_invalid_source_or_shortage_leaves_no_output(self):
        invalids = [
            b"not JSON\n", b"[]\n", b'{"content":2,"moderation":{"A":1}}\n',
            b'{"content":"x","moderation":{}}\n', b'{"content":"x","moderation":{"A":NaN}}\n',
            b'{"content":"x","moderation":{"A":true}}\n', b'{"content":"x","moderation":{"A":0}}\n',
            b'{"content":"x","moderation":{"A":-1}}\n', b'{"content":"x","moderation":{"A":2}}\n',
            b'{"content":"x","moderation":{"A":1},"sampling":{}}\n',
        ]
        for invalid in invalids:
            with self.subTest(invalid=invalid), TemporaryDirectory() as directory:
                source, output = Path(directory) / "source.jsonl", Path(directory) / "sample.jsonl"
                source.write_bytes(invalid)
                with self.assertRaises(ValueError):
                    guided_sample_dataset(source, output, {"A": 1})
                self.assertFalse(output.exists())
                self.assertFalse(Path(str(output) + ".manifest.json").exists())
                self.assertFalse(list(Path(directory).glob("guided-sample-*")))
        with TemporaryDirectory() as directory:
            source, output = Path(directory) / "source.jsonl", Path(directory) / "sample.jsonl"
            self.write_rows(source, self.make_rows())
            with self.assertRaisesRegex(ValueError, "has 30 rows"):
                guided_sample_dataset(source, output, {"A": 31})
            with self.assertRaisesRegex(ValueError, "has 0 rows"):
                guided_sample_dataset(source, output, {"absent": 1})
            self.assertFalse(output.exists())

    def test_configuration_and_source_path_validation(self):
        with TemporaryDirectory() as directory:
            source, output = Path(directory) / "source.jsonl", Path(directory) / "sample.jsonl"
            self.write_rows(source, self.make_rows())
            for quotas in ({}, {"A": 0}, {"A": -1}, {"A": True}, {"A": 1.5}, {"": 1}):
                with self.subTest(quotas=quotas), self.assertRaises(ValueError):
                    guided_sample_dataset(source, output, quotas)
            for options in ({"seed": True}, {"ambiguous_fraction": math.nan}, {"long_fraction": -0.1},
                            {"ambiguous_fraction": 0.8, "long_fraction": 0.3},
                            {"metadata_field": "content"}, {"score_field": ""}):
                with self.subTest(options=options), self.assertRaises(ValueError):
                    guided_sample_dataset(source, output, {"A": 1}, **options)
            with self.assertRaises(ValueError):
                guided_sample_dataset(source, source, {"A": 1})

    def test_custom_fields_and_failed_publication(self):
        with TemporaryDirectory() as directory:
            source, output = Path(directory) / "source.jsonl", Path(directory) / "sample.jsonl"
            self.write_rows(source, [{"text": "🌍", "scores": {"A": 1.0}, "sampling": "preserved"}])
            options = {"text_field": "text", "score_field": "scores", "metadata_field": "selection"}
            guided_sample_dataset(source, output, {"A": 1}, **options)
            selected = self.load_rows(output)[0]
            self.assertEqual(selected["sampling"], "preserved")
            self.assertEqual(selected["selection"]["content_characters"], 1)
            other = Path(directory) / "failed.jsonl"
            from distill_jev.datasets import sampling
            original_link = sampling.os.link
            calls = []

            def link(source_path, target):
                calls.append(target)
                if len(calls) == 2:
                    raise OSError("manifest publication failed")
                original_link(source_path, target)

            with patch.object(sampling.os, "link", side_effect=link):
                with self.assertRaisesRegex(OSError, "manifest publication"):
                    guided_sample_dataset(source, other, {"A": 1}, **options)
            self.assertFalse(other.exists())
            self.assertFalse(Path(str(other) + ".manifest.json").exists())
