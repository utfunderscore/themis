import json
import unittest
from contextlib import chdir
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import call, patch

import polars as pl

from themis.datasets.deduplication import deduplicate_chats
from themis.pipeline import generate_review_dataset, pipeline


class CombinedPipelineTests(unittest.TestCase):
    def test_review_dataset_uses_each_prediction_and_preserves_context(self):
        rows = [
            {"state": "first chat", "images": ["data:image/png;base64,abc"],
             "decision": {"answer": {"probabilities": {"none": 0.1, "insult": 0.9}}}},
            {"state": "second chat", "images": [],
             "decision": {"answer": {"probabilities": {"none": 0.9, "insult": 0.1}}}},
        ]
        with TemporaryDirectory() as directory:
            source, destination = Path(directory) / "scores.jsonl", Path(directory) / "review.jsonl"
            source.write_text("".join(json.dumps(row) + "\n" for row in rows))
            self.assertEqual(generate_review_dataset(source=source, destination=destination), destination)
            prepared = [json.loads(line) for line in destination.read_text().splitlines()]
            for row, original, category in zip(prepared, rows, ("insult", "none"), strict=True):
                self.assertIn(f"classified as {category}.", row["question"]["instructions"])
                self.assertEqual(list(row["question"]["criteria"]), ["allow", "block_and_review"])
                self.assertEqual(row["source"], original)
                self.assertNotIn("decision", row)
            self.assertEqual(prepared[0]["images"], rows[0]["images"])
            self.assertEqual(prepared[1]["state"], "second chat")
            source.write_text(json.dumps({"content": "legacy chat", "decision": rows[1]["decision"]}))
            generate_review_dataset(source=source, destination=destination)
            legacy = json.loads(destination.read_text())
            self.assertEqual(legacy["state"], "legacy chat")
            self.assertIn("classified as none.", legacy["question"]["instructions"])

    def setUp(self):
        # Pipeline tests must never start paid Perplexity inference.
        patcher = patch("themis.pipeline.predict_dataset")
        self.predict = patcher.start()
        self.addCleanup(patcher.stop)
        patcher = patch("themis.pipeline.prepare_dataset")
        self.prepare = patcher.start()
        self.addCleanup(patcher.stop)
        patcher = patch("themis.pipeline.generate_review_dataset")
        self.review = patcher.start()
        self.addCleanup(patcher.stop)
        patcher = patch("themis.pipeline.train_and_export")
        self.training = patcher.start()
        self.addCleanup(patcher.stop)

    def test_pipeline_scores_sample_and_reuses_completed_predictions(self):
        from themis.models.decision.perplexity.questions import CATEGORY_QUESTION

        with TemporaryDirectory() as directory, chdir(directory):
            source = Path("cache/combined_chats_moderated.jsonl")
            source.parent.mkdir()
            source.touch()
            destination = Path("cache/combined_chats_inference_sample_100k_decisions.jsonl")
            review_inputs = Path("cache/combined_chats_inference_sample_100k_review_inputs.jsonl")
            review_decisions = Path("cache/combined_chats_inference_sample_100k_review_decisions.jsonl")
            with patch("themis.pipeline.guided_sample_dataset"):
                pipeline()
                self.prepare.assert_called_once_with(
                    source=Path("cache/combined_chats_inference_sample_100k.jsonl"),
                    destination=Path("cache/combined_chats_inference_sample_100k_decision_inputs.jsonl"),
                    question=CATEGORY_QUESTION,
                )
                self.assertEqual(self.predict.call_args_list, [
                    call(
                        input_path=Path("cache/combined_chats_inference_sample_100k_decision_inputs.jsonl"),
                        output_path=destination, batch_size=32, rows_per_shard=1024,
                        max_containers=4,
                    ),
                    call(
                        input_path=review_inputs, output_path=review_decisions,
                        batch_size=32, rows_per_shard=1024, max_containers=4,
                    ),
                ])
                self.review.assert_called_once_with(
                    source=destination,
                    destination=review_inputs,
                )
                self.training.assert_called_once_with(
                    source=review_decisions,
                    training_data_directory="cache/deberta/review-splits",
                    onnx_output_directory=None,
                )
                destination.write_text("completed output")
                self.predict.reset_mock()
                self.review.reset_mock()
                pipeline()
                self.predict.assert_called_once_with(
                    input_path=review_inputs, output_path=review_decisions,
                    batch_size=32, rows_per_shard=1024, max_containers=4,
                )
                self.review.assert_called_once()
                self.assertEqual(destination.read_text(), "completed output")

    def test_inference_failure_does_not_report_completion(self):
        with TemporaryDirectory() as directory, chdir(directory):
            source = Path("cache/combined_chats_moderated.jsonl")
            source.parent.mkdir()
            source.touch()
            self.predict.side_effect = RuntimeError("inference failed")
            with (patch("themis.pipeline.guided_sample_dataset"),
                  patch("builtins.print") as output):
                with self.assertRaisesRegex(RuntimeError, "inference failed"):
                    pipeline()
                self.review.assert_not_called()
                self.training.assert_not_called()
                self.assertFalse(any(call.args == ("Pipeline complete.",) for call in output.call_args_list))

    def test_pipeline_scores_review_from_cached_predictions(self):
        with TemporaryDirectory() as directory, chdir(directory):
            source = Path("cache/combined_chats_moderated.jsonl")
            source.parent.mkdir()
            source.touch()
            decisions = Path("cache/combined_chats_inference_sample_100k_decisions.jsonl")
            original = {
                "content": "game chat",
                "decision": {"answer": {"probabilities": {"none": 0.8, "threat": 0.2}}},
            }
            decisions.write_text(json.dumps(original) + "\n")
            self.review.side_effect = generate_review_dataset

            def score_review(input_path, output_path, **options):
                row = json.loads(input_path.read_text())
                self.assertNotIn("decision", row)
                self.assertEqual(list(row["question"]["criteria"]), ["allow", "block_and_review"])
                row["decision"] = {"answer": {"choice": "allow"}}
                output_path.write_text(json.dumps(row) + "\n")
                return output_path

            self.predict.side_effect = score_review
            with patch("themis.pipeline.guided_sample_dataset"):
                pipeline()
            output = Path("cache/combined_chats_inference_sample_100k_review_inputs.jsonl")
            review_decisions = Path("cache/combined_chats_inference_sample_100k_review_decisions.jsonl")
            self.predict.assert_called_once_with(
                input_path=output, output_path=review_decisions,
                batch_size=32, rows_per_shard=1024, max_containers=4,
            )
            row = json.loads(output.read_text())
            self.assertEqual(row["state"], "game chat")
            self.assertIn("classified as none.", row["question"]["instructions"])
            scored = json.loads(review_decisions.read_text())
            self.assertEqual(scored["decision"]["answer"]["choice"], "allow")
            self.assertEqual(scored["source"], original)
            self.assertEqual(json.loads(decisions.read_text()), original)

    def test_pipeline_reuses_completed_review_without_review_inputs(self):
        with TemporaryDirectory() as directory, chdir(directory):
            source = Path("cache/combined_chats_moderated.jsonl")
            source.parent.mkdir()
            source.touch()
            Path("cache/combined_chats_inference_sample_100k_decisions.jsonl").touch()
            destination = Path("cache/combined_chats_inference_sample_100k_review_decisions.jsonl")
            destination.write_text("completed review output")
            with patch("themis.pipeline.guided_sample_dataset"):
                pipeline()
            self.prepare.assert_not_called()
            self.review.assert_not_called()
            self.predict.assert_not_called()
            self.assertEqual(destination.read_text(), "completed review output")
            self.assertFalse(Path("cache/combined_chats_inference_sample_100k_review_inputs.jsonl").exists())

    def test_review_inference_failure_can_retry_with_identical_inputs(self):
        with TemporaryDirectory() as directory, chdir(directory):
            source = Path("cache/combined_chats_moderated.jsonl")
            source.parent.mkdir()
            source.touch()
            decisions = Path("cache/combined_chats_inference_sample_100k_decisions.jsonl")
            decisions.write_text(json.dumps({
                "content": "game chat",
                "decision": {"answer": {"probabilities": {"none": 1.0}}},
            }) + "\n")
            self.review.side_effect = generate_review_dataset
            self.predict.side_effect = RuntimeError("review inference failed")
            with (patch("themis.pipeline.guided_sample_dataset"),
                  patch("builtins.print") as output):
                with self.assertRaisesRegex(RuntimeError, "review inference failed"):
                    pipeline()
                self.assertFalse(any(call.args == ("Pipeline complete.",) for call in output.call_args_list))
                review_inputs = Path("cache/combined_chats_inference_sample_100k_review_inputs.jsonl")
                before = review_inputs.read_bytes()
                first_call = self.predict.call_args
                self.predict.side_effect = None
                pipeline()
                self.assertEqual(self.predict.call_args, first_call)
                self.assertEqual(review_inputs.read_bytes(), before)
                self.prepare.assert_not_called()
                output.assert_called_with("Pipeline complete.", flush=True)

    def test_review_failure_does_not_report_completion(self):
        with TemporaryDirectory() as directory, chdir(directory):
            source = Path("cache/combined_chats_moderated.jsonl")
            source.parent.mkdir()
            source.touch()
            self.review.side_effect = ValueError("invalid predictions")
            with (patch("themis.pipeline.guided_sample_dataset"),
                  patch("builtins.print") as output):
                with self.assertRaisesRegex(ValueError, "invalid predictions"):
                    pipeline()
                self.assertFalse(any(call.args == ("Pipeline complete.",) for call in output.call_args_list))

    def test_training_failure_does_not_report_completion(self):
        with TemporaryDirectory() as directory, chdir(directory):
            self.training.side_effect = RuntimeError("training failed")
            with (patch("themis.pipeline.Path.is_file", return_value=True),
                  patch("themis.pipeline.guided_sample_dataset"),
                  patch("builtins.print") as output):
                with self.assertRaisesRegex(RuntimeError, "training failed"):
                    pipeline()
                self.assertFalse(any(call.args == ("Pipeline complete.",) for call in output.call_args_list))

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
                patch("themis.pipeline.download_minecraft_chats", return_value=minecraft) as first,
                patch("themis.pipeline.download_2b2t_chats", return_value=two_b_two_t) as second,
                patch("themis.pipeline.moderate_dataset") as moderate,
                patch("themis.pipeline.guided_sample_dataset") as sample,
            ):
                self.assertIsNone(pipeline())
                moderate.assert_called_once_with(
                    input_path=Path("cache/combined_chats_deduplicated.jsonl"),
                    output_path=Path("cache/combined_chats_moderated.jsonl"),
                    text_field="content", batch_size=128,
                )
                sample.assert_called_once_with(
                    source=Path("cache/combined_chats_moderated.jsonl"),
                    destination=Path("cache/combined_chats_inference_sample_100k.jsonl"),
                    quotas={"OK": 49_911, "H": 20_000, "V": 10_000, "S": 8_000,
                     "SH": 7_000, "HR": 5_088, "H2": 1}, seed=42,
                )
            first.assert_called_once_with(output_path=Path("cache/minecraft_chats.jsonl"), force=False)
            second.assert_called_once_with(output_path=Path("cache/2b2t_chats.csv"), force=False)
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
                patch("themis.pipeline.download_minecraft_chats") as first,
                patch("themis.pipeline.download_2b2t_chats") as second,
                patch("themis.pipeline.deduplicate_chats") as deduplicate,
                patch("themis.pipeline.moderate_dataset") as moderate,
                patch("themis.pipeline.guided_sample_dataset") as sample,
            ):
                self.assertIsNone(pipeline())
                first.assert_not_called()
                second.assert_not_called()
                deduplicate.assert_not_called()
                moderate.assert_not_called()
                self.assertEqual(sample.call_args.kwargs["source"], destination)
            self.assertEqual(destination.read_text(), contents)

    def test_pipeline_reuses_existing_input_for_resume(self) -> None:
        with TemporaryDirectory() as directory, chdir(directory):
            source = Path("cache/combined_chats_deduplicated.jsonl")
            source.parent.mkdir()
            source.write_text('{"content":"cached"}\n')
            with (
                patch("themis.pipeline.download_minecraft_chats") as download,
                patch("themis.pipeline.deduplicate_chats") as deduplicate,
                patch("themis.pipeline.moderate_dataset") as moderate,
                patch("themis.pipeline.guided_sample_dataset") as sample,
            ):
                self.assertIsNone(pipeline())
                download.assert_not_called()
                deduplicate.assert_not_called()
                self.assertEqual(moderate.call_args.kwargs["input_path"], source)
                self.assertEqual(sample.call_args.kwargs["source"], Path("cache/combined_chats_moderated.jsonl"))

    def test_sampling_is_not_started_when_moderation_fails(self):
        with TemporaryDirectory() as directory, chdir(directory):
            source = Path("cache/combined_chats_deduplicated.jsonl")
            source.parent.mkdir()
            source.write_text('{"content":"cached"}\n')
            with (patch("themis.pipeline.moderate_dataset", side_effect=RuntimeError("failed")),
                  patch("themis.pipeline.guided_sample_dataset") as sample):
                with self.assertRaises(RuntimeError):
                    pipeline()
                sample.assert_not_called()

    def test_approved_sample_quotas(self):
        with (patch("themis.pipeline.Path.is_file", return_value=True),
              patch("themis.pipeline.guided_sample_dataset") as sample):
            pipeline()
            self.assertEqual(sum(sample.call_args.kwargs["quotas"].values()), 100_000)

    def test_pipeline_builds_and_reuses_local_sample_from_completed_moderation(self):
        with TemporaryDirectory() as directory, chdir(directory):
            source = Path("cache/combined_chats_moderated.jsonl")
            source.parent.mkdir()
            source.write_text(
                '{"content":"one","moderation":{"A":0.7,"B":0.3}}\n'
                '{"content":"two","moderation":{"A":0.6,"B":0.4}}\n'
            )
            from themis.datasets.sampling import guided_sample_dataset

            def sample_small(source, destination, quotas, **options):
                return guided_sample_dataset(source, destination, {"A": 1}, **options)

            with (patch("themis.pipeline.guided_sample_dataset", side_effect=sample_small),
                  patch("themis.pipeline.download_minecraft_chats") as download,
                  patch("themis.pipeline.moderate_dataset") as moderate):
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
            with (patch("themis.pipeline.guided_sample_dataset") as sample,
                  patch("builtins.print") as output):
                self.assertIsNone(pipeline())
                output.assert_called_with("Pipeline complete.", flush=True)
                output.reset_mock()
                sample.side_effect = RuntimeError("sampling failed")
                with self.assertRaisesRegex(RuntimeError, "sampling failed"):
                    pipeline()
                self.assertFalse(any(call.args == ("Pipeline complete.",) for call in output.call_args_list))
