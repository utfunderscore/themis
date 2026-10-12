import json
import unittest
from contextlib import chdir
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import polars as pl

from distill_jev.datasets.deduplication import deduplicate_chats
from distill_jev.pipeline import pipeline


class CombinedPipelineTests(unittest.TestCase):
    def test_deduplicates_lazy_frame(self) -> None:
        chats = pl.DataFrame({"content": ["hello", "hello", "Hello", ""]}).lazy()
        with TemporaryDirectory() as directory:
            destination = Path(directory) / "chats.jsonl"
            self.assertEqual(deduplicate_chats(chats, destination), destination)
            self.assertEqual(
                {json.loads(line)["content"] for line in destination.read_text().splitlines()},
                {"hello", "Hello", ""},
            )

    def test_pipeline_combines_and_deduplicates_both_sources(self) -> None:
        minecraft = pl.DataFrame({
            "username": ["minecraft", "minecraft"],
            "content": ["shared", "minecraft only"],
            "date": ["2020", "2020"],
        }).lazy()
        two_b_two_t = pl.DataFrame({
            "username": ["2b2t", "2b2t", "2b2t"],
            "content": ["shared", "2b2t only", "2b2t only"],
            "date": ["2025", "2025", "2025"],
        }).lazy()
        with TemporaryDirectory() as directory, chdir(directory):
            with (
                patch("distill_jev.pipeline.download_minecraft_chats", return_value=minecraft) as first,
                patch("distill_jev.pipeline.download_2b2t_chats", return_value=two_b_two_t) as second,
                patch("distill_jev.pipeline.moderate_dataset") as moderate,
                patch("distill_jev.pipeline.guided_sample_dataset") as sample,
            ):
                self.assertIsNone(pipeline())
                moderate.assert_called_once_with(
                    Path("cache/combined_chats_deduplicated.jsonl"),
                    Path("cache/combined_chats_moderated.jsonl"),
                    text_field="content", batch_size=128,
                )
                sample.assert_called_once_with(
                    Path("cache/combined_chats_moderated.jsonl"),
                    Path("cache/combined_chats_inference_sample_100k.jsonl"),
                    {"OK": 49_911, "H": 20_000, "V": 10_000, "S": 8_000,
                     "SH": 7_000, "HR": 5_088, "H2": 1}, seed=42,
                )
            first.assert_called_once_with(Path("cache/minecraft_chats.jsonl"), force=False)
            second.assert_called_once_with(Path("cache/2b2t_chats.csv"), force=False)
            output = Path("cache/combined_chats_deduplicated.jsonl")
            rows = [json.loads(line) for line in output.read_text().splitlines()]
            self.assertEqual(len(rows), 3)
            self.assertEqual({row["content"] for row in rows}, {"shared", "minecraft only", "2b2t only"})
            self.assertEqual(next(row["username"] for row in rows if row["content"] == "shared"), "minecraft")

    def test_pipeline_reuses_completed_output_without_input(self) -> None:
        with TemporaryDirectory() as directory, chdir(directory):
            destination = Path("cache/combined_chats_moderated.jsonl")
            destination.parent.mkdir()
            contents = '{"content":"cached","moderation":{"OK":1.0}}\n'
            destination.write_text(contents)
            with (
                patch("distill_jev.pipeline.download_minecraft_chats") as first,
                patch("distill_jev.pipeline.download_2b2t_chats") as second,
                patch("distill_jev.pipeline.deduplicate_chats") as deduplicate,
                patch("distill_jev.pipeline.moderate_dataset") as moderate,
                patch("distill_jev.pipeline.guided_sample_dataset") as sample,
            ):
                self.assertIsNone(pipeline())
                first.assert_not_called()
                second.assert_not_called()
                deduplicate.assert_not_called()
                moderate.assert_not_called()
                self.assertEqual(sample.call_args.args[0], destination)
            self.assertEqual(destination.read_text(), contents)

    def test_pipeline_reuses_existing_input_for_resume(self) -> None:
        with TemporaryDirectory() as directory, chdir(directory):
            source = Path("cache/combined_chats_deduplicated.jsonl")
            source.parent.mkdir()
            source.write_text('{"content":"cached"}\n')
            with (
                patch("distill_jev.pipeline.download_minecraft_chats") as download,
                patch("distill_jev.pipeline.deduplicate_chats") as deduplicate,
                patch("distill_jev.pipeline.moderate_dataset") as moderate,
                patch("distill_jev.pipeline.guided_sample_dataset") as sample,
            ):
                self.assertIsNone(pipeline())
                download.assert_not_called()
                deduplicate.assert_not_called()
                self.assertEqual(moderate.call_args.args[0], source)
                self.assertEqual(sample.call_args.args[0], Path("cache/combined_chats_moderated.jsonl"))

    def test_sampling_is_not_started_when_moderation_fails(self):
        with TemporaryDirectory() as directory, chdir(directory):
            source = Path("cache/combined_chats_deduplicated.jsonl")
            source.parent.mkdir()
            source.write_text('{"content":"cached"}\n')
            with (patch("distill_jev.pipeline.moderate_dataset", side_effect=RuntimeError("failed")),
                  patch("distill_jev.pipeline.guided_sample_dataset") as sample):
                with self.assertRaises(RuntimeError):
                    pipeline()
                sample.assert_not_called()

    def test_approved_sample_quotas(self):
        with (patch("distill_jev.pipeline.Path.is_file", return_value=True),
              patch("distill_jev.pipeline.guided_sample_dataset") as sample):
            pipeline()
            self.assertEqual(sum(sample.call_args.args[2].values()), 100_000)

    def test_pipeline_builds_and_reuses_local_sample_from_completed_moderation(self):
        with TemporaryDirectory() as directory, chdir(directory):
            source = Path("cache/combined_chats_moderated.jsonl")
            source.parent.mkdir()
            source.write_text(
                '{"content":"one","moderation":{"A":0.7,"B":0.3}}\n'
                '{"content":"two","moderation":{"A":0.6,"B":0.4}}\n'
            )
            from distill_jev.datasets.sampling import guided_sample_dataset

            def sample_small(source, destination, quotas, **options):
                return guided_sample_dataset(source, destination, {"A": 1}, **options)

            with (patch("distill_jev.pipeline.guided_sample_dataset", side_effect=sample_small),
                  patch("distill_jev.pipeline.download_minecraft_chats") as download,
                  patch("distill_jev.pipeline.moderate_dataset") as moderate):
                self.assertIsNone(pipeline())
                output = Path("cache/combined_chats_inference_sample_100k.jsonl")
                self.assertEqual(len(output.read_text().splitlines()), 1)
                timestamp = output.stat().st_mtime_ns
                self.assertIsNone(pipeline())
                self.assertEqual(timestamp, output.stat().st_mtime_ns)
                self.assertTrue(Path(str(output) + ".manifest.json").is_file())
                download.assert_not_called()
                moderate.assert_not_called()

    def test_pipeline_reports_completion_only_after_sampling_succeeds(self):
        with TemporaryDirectory() as directory, chdir(directory):
            source = Path("cache/combined_chats_moderated.jsonl")
            source.parent.mkdir()
            source.write_text('{"content":"cached","moderation":{"OK":1.0}}\n')
            with (patch("distill_jev.pipeline.guided_sample_dataset") as sample,
                  patch("builtins.print") as output):
                self.assertIsNone(pipeline())
                output.assert_called_with("Pipeline complete.", flush=True)
                output.reset_mock()
                sample.side_effect = RuntimeError("sampling failed")
                with self.assertRaisesRegex(RuntimeError, "sampling failed"):
                    pipeline()
                self.assertFalse(any(call.args == ("Pipeline complete.",) for call in output.call_args_list))
