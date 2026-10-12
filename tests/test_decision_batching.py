"""CPU-only scheduling/experiment tests; Modal calls are mocked or never made."""

import json
import unittest
from contextlib import nullcontext
from functools import wraps
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from distill_jev.models.decision.perplexity.batching import (
    DEFAULT_BATCH_SIZE, DEFAULT_PADDED_TOKEN_BUDGET, plan_batches,
    predict_indexed_batch, predict_length_batched, rendered_input_lengths,
)
from distill_jev.models.decision.perplexity.experiment import (
    _parity_summary, benchmark_batched_model, compare_predictions, validate_experiment_settings,
)


class DecisionBatchingTests(unittest.TestCase):
    def test_exact_native_preparation_includes_full_input_and_rejects_overflow(self):
        rows = [{"state": "a", "images": ["data:image/png;base64,test"]}, {"state": "b"}]
        model = SimpleNamespace(prepare=MagicMock(side_effect=[
            SimpleNamespace(input_tokens=8000), SimpleNamespace(input_tokens=8192)]))
        self.assertEqual(rendered_input_lengths(model, rows), [8000, 8192])
        self.assertEqual(model.prepare.call_args_list[0].args, ([rows[0]],))
        self.assertEqual(model.prepare.call_args_list[0].kwargs, {"max_length": 8192})
        model.prepare.side_effect = None
        model.prepare.return_value = SimpleNamespace(input_tokens=8193)
        with self.assertRaisesRegex(ValueError, "invalid full input length"):
            rendered_input_lengths(model, rows)
        model.prepare.side_effect = ValueError("native overflow; no input was truncated")
        with self.assertRaisesRegex(ValueError, "no input was truncated"):
            rendered_input_lengths(model, rows)

    def test_stable_sort_token_budget_and_original_indices(self):
        self.assertEqual(plan_batches([3, 1, 2, 1], 2), [[1, 3], [2, 0]])
        self.assertEqual(plan_batches([3, 1, 2, 1], 2, length_sorted=False), [[0, 1], [2, 3]])
        plan = plan_batches([8192, 1, 8192, 1, 5000], 32, max_padded_tokens=8192)
        self.assertEqual(plan, [[1, 3], [4], [0], [2]])
        self.assertEqual(sorted(index for batch in plan for index in batch), list(range(5)))
        self.assertEqual([len(batch) for batch in plan_batches([100] * 65, 32)], [32, 32, 1])
        for args in (([8193], 8), ([0], 8), ([True], 8), ([1], 33), ([1], True)):
            with self.assertRaises(ValueError):
                plan_batches(*args)
        with self.assertRaises(ValueError):
            plan_batches([1], 8, max_padded_tokens=8191)

    def test_oom_splitting_and_strict_output_count(self):
        cuda = SimpleNamespace(OutOfMemoryError=MemoryError, synchronize=MagicMock(), empty_cache=MagicMock())
        torch = SimpleNamespace(cuda=cuda)
        rows = [{"state": index} for index in range(8)]

        def predict(batch, batch_size):
            if batch_size > 2:
                raise MemoryError("simulated CUDA OOM")
            return [[row["state"]] for row in batch]

        model = SimpleNamespace(predict=MagicMock(side_effect=predict))
        stats = {"successful_batch_sizes": [], "batch_latencies_seconds": [], "oom_splits": []}
        result = predict_indexed_batch(model, rows, list(reversed(range(8))), torch, stats)
        self.assertEqual(result, [(index, [index]) for index in reversed(range(8))])
        self.assertEqual(stats["successful_batch_sizes"], [2] * 4)
        self.assertEqual(len(stats["oom_splits"]), 3)
        self.assertEqual(cuda.empty_cache.call_count, 3)
        cuda.empty_cache.reset_mock()
        predict_indexed_batch(model, rows, [0, 1], torch, stats)
        cuda.empty_cache.assert_not_called()
        model.predict.side_effect = MemoryError("singleton OOM")
        with self.assertRaisesRegex(MemoryError, "singleton"):
            predict_indexed_batch(model, rows, [0], torch, stats)
        model.predict.side_effect = RuntimeError("not CUDA OOM")
        with self.assertRaisesRegex(RuntimeError, "not CUDA"):
            predict_indexed_batch(model, rows, [0, 1], torch, stats)
        model.predict.side_effect = lambda batch, batch_size: []
        with self.assertRaises(ValueError):
            predict_indexed_batch(model, rows, [0], torch, stats)

    def test_serving_default_sorts_thirty_two_and_restores_exact_scores(self):
        self.assertEqual(DEFAULT_BATCH_SIZE, 32)
        self.assertEqual(DEFAULT_PADDED_TOKEN_BUDGET, 16384)
        rows = [{"state": str(index), "question": {"type": "noul"}} for index in range(65)]
        expected = [[index / 100, 1 - index / 100] for index in range(65)]
        calls = []

        def predict(batch, batch_size):
            indices = [int(row["state"]) for row in batch]
            calls.append((indices, batch_size))
            return [expected[index] for index in indices]

        model = SimpleNamespace(
            prepare=lambda batch, max_length: SimpleNamespace(input_tokens=300 - int(batch[0]["state"])),
            predict=predict,
        )
        cuda = SimpleNamespace(OutOfMemoryError=MemoryError, synchronize=MagicMock(), empty_cache=MagicMock())
        with patch("distill_jev.models.decision.perplexity.batching.import_module",
                   return_value=SimpleNamespace(cuda=cuda)):
            probabilities, stats = predict_length_batched(model, rows)
        self.assertEqual([size for _, size in calls], [32, 32, 1])
        self.assertEqual([index for indices, _ in calls for index in indices], list(reversed(range(65))))
        self.assertEqual(probabilities, expected)
        self.assertTrue(all(left is right for left, right in zip(probabilities, expected, strict=True)))
        self.assertEqual(stats["runtime_batch_cap"], 32)
        cuda.empty_cache.assert_not_called()

    def test_serving_token_guard_overflow_and_oom_cap(self):
        cuda = SimpleNamespace(OutOfMemoryError=MemoryError, synchronize=MagicMock(), empty_cache=MagicMock())
        torch = SimpleNamespace(cuda=cuda)
        target = "distill_jev.models.decision.perplexity.batching.import_module"
        rows = [{"state": str(index), "question": {"type": "noul"}} for index in range(65)]
        model = SimpleNamespace(
            prepare=MagicMock(return_value=SimpleNamespace(input_tokens=8192)),
            predict=MagicMock(side_effect=lambda batch, batch_size: [[int(row["state"])] for row in batch]),
        )
        with patch(target, return_value=torch):
            probabilities, stats = predict_length_batched(model, rows)
        self.assertEqual(probabilities, [[index] for index in range(65)])
        self.assertEqual(set(stats["successful_batch_sizes"]), {1, 2})
        # An overflow anywhere rejects the entire call before any model forward.
        model.predict.reset_mock()
        model.prepare.side_effect = [SimpleNamespace(input_tokens=10), SimpleNamespace(input_tokens=8193)]
        with patch(target, return_value=torch):
            with self.assertRaises(ValueError):
                predict_length_batched(model, rows)
        model.predict.assert_not_called()
        model.prepare.side_effect = None
        model.prepare.return_value = SimpleNamespace(input_tokens=250)
        sizes = []

        def predict(batch, batch_size):
            sizes.append(batch_size)
            if batch_size > 16:
                raise MemoryError("simulated CUDA OOM")
            return [[int(row["state"])] for row in batch]

        model.predict.side_effect = predict
        with patch(target, return_value=torch):
            probabilities, stats = predict_length_batched(model, rows)
        self.assertEqual(probabilities, [[index] for index in range(65)])
        self.assertEqual(sizes, [32, 16, 16, 16, 16, 1])
        self.assertEqual(stats["runtime_batch_cap"], 16)
        self.assertEqual(len(stats["oom_splits"]), 1)
        model.predict.side_effect = MemoryError("singleton OOM")
        with patch(target, return_value=torch):
            with self.assertRaisesRegex(MemoryError, "singleton"):
                predict_length_batched(model, rows[:1])

    def test_serving_invalid_batch_size_rejected_before_preparation(self):
        model = SimpleNamespace(prepare=MagicMock(), predict=MagicMock())
        for size in (0, 33, True):
            with self.assertRaises(ValueError):
                predict_length_batched(model, [{"state": "example"}], batch_size=size)
        model.prepare.assert_not_called()
        model.predict.assert_not_called()

    def test_experiment_settings_and_preflight_validation(self):
        settings = {"batch_sizes": [8, 16, 32], "trials": 3, "min_trial_seconds": 15,
                    "max_trial_passes": 20, "max_padded_tokens": 16384}
        validate_experiment_settings(settings)
        for key, bad in (("batch_sizes", [8, 8]), ("batch_sizes", [64]), ("trials", 0),
                         ("trials", True), ("min_trial_seconds", float("nan")),
                         ("max_trial_passes", 101), ("max_padded_tokens", 100)):
            with self.assertRaises(ValueError):
                validate_experiment_settings({**settings, key: bad})
        from distill_jev.models.decision.perplexity import benchmark
        with patch.object(benchmark.optimization_app, "run") as run:
            with self.assertRaises(ValueError):
                benchmark.run_optimization("unused", "unused", batch_sizes=(64,))
            run.assert_not_called()

    def test_candidate_order_changes_rejected_before_loading_dependencies(self):
        rows = [{"state": "same", "question": {"type": "choice", "criteria": {"a": "A", "b": "B"}}}
                for _ in range(220)]
        rows[-1] = {"state": "same", "question": {"type": "choice", "criteria": {"b": "B", "a": "A"}}}
        settings = {"batch_sizes": [8], "trials": 1, "min_trial_seconds": 0,
                    "max_trial_passes": 1, "max_padded_tokens": 16384}
        with patch("distill_jev.models.decision.perplexity.experiment.import_module") as imports:
            with self.assertRaisesRegex(ValueError, "candidate order"):
                benchmark_batched_model(None, None, rows, 0, model_id="test",
                                        model_revision="pinned", settings=settings)
            imports.assert_not_called()

    def test_parity_summary_counts_flip_in_any_pass_and_excludes_invalid_rows(self):
        reference = [{"probabilities": [0.75, 0.25], "answer": "a"}] * 3
        first = compare_predictions(reference, [
            {"probabilities": [0.25, 0.75], "answer": "b"}, reference[1], reference[2]])
        second = compare_predictions(reference, [
            reference[0], {"error": "nonfinite"}, reference[2]])
        summary = _parity_summary([first, second], 3)
        self.assertEqual(summary["checked_passes"], 2)
        self.assertEqual(summary["changed_answer_indices"], [0])
        self.assertEqual(summary["invalid_pair_indices"], [1])
        self.assertEqual(summary["compared_valid_rows"], 2)
        self.assertEqual(summary["answer_agreement_rows"], 1)
        self.assertEqual(summary["answer_agreement_fraction"], 0.5)
        self.assertEqual(summary["max_absolute_probability_difference"], 0.5)
        self.assertAlmostEqual(summary["mean_absolute_probability_difference"], 0.1)

    def run_fake_experiment(self, *, fail_optimized=False, oom_limit=None, min_seconds=3,
                            max_passes=10, invalid=False):
        def reference():
            pass
        reference.__module__ = "fake_qwen"

        @wraps(reference)
        def selected():
            pass

        qwen = SimpleNamespace(__name__="fake_qwen", causal_conv1d_fn=selected)
        cuda = SimpleNamespace(
            synchronize=MagicMock(), empty_cache=MagicMock(), reset_peak_memory_stats=MagicMock(),
            max_memory_allocated=lambda: 1024, max_memory_reserved=lambda: 2048,
            get_device_name=lambda _: "fake H100",
            get_device_properties=lambda _: SimpleNamespace(total_memory=4096),
            OutOfMemoryError=MemoryError,
        )
        torch = SimpleNamespace(cuda=cuda, bfloat16="BF16", version=SimpleNamespace(cuda="13.0"))
        clock = [0.0]

        def predict(batch, batch_size):
            clock[0] += 0.2
            if qwen.causal_conv1d_fn is not reference:
                if fail_optimized:
                    raise RuntimeError("optimized failed")
                if oom_limit is not None and batch_size > oom_limit:
                    raise MemoryError("simulated CUDA OOM")
            return [[float("nan"), float("inf")] if invalid else
                    [0.75 + int(row["state"]) / 10000, 0.25 - int(row["state"]) / 10000]
                    for row in batch]

        def answer(question, scores):
            if invalid:
                raise ValueError("nonfinite")
            return {"type": "choice", "choice": "a"}

        model = SimpleNamespace(
            prepare=lambda batch, max_length: SimpleNamespace(input_tokens=240 + int(batch[0]["state"]) % 35),
            predict=predict, backbone=SimpleNamespace(parameters=lambda: iter([SimpleNamespace(dtype="BF16")])),
        )
        modules = {"torch": torch, "transformers.models.qwen3_5.modeling_qwen3_5": qwen,
                   "causal_conv1d": SimpleNamespace(causal_conv1d_fn=lambda: None)}
        snapshots = []
        target = "distill_jev.models.decision.perplexity.experiment"
        error = None
        report = None
        with (patch(f"{target}.import_module", side_effect=modules.__getitem__),
              patch(f"{target}.metadata.version", return_value="test"),
              patch(f"{target}.perf_counter", side_effect=lambda: clock[0]),
              patch(f"{target}.log"), patch(f"{target}.stage", side_effect=lambda _: nullcontext())):
            try:
                report = benchmark_batched_model(
                    model, answer, [{"state": str(index), "question": {"type": "choice", "criteria": {"a": "A", "b": "B"}}}
                                    for index in range(220)], 1.0, model_id="test", model_revision="pinned",
                    settings={"batch_sizes": [8, 16, 32], "trials": 2, "min_trial_seconds": min_seconds,
                              "max_trial_passes": max_passes, "max_padded_tokens": 16384},
                    persist=lambda value: snapshots.append(json.loads(json.dumps(value, allow_nan=False))),
                    provenance={"seed": 42},
                )
            except RuntimeError as exc:
                error = exc
        self.assertIs(qwen.causal_conv1d_fn, selected)
        return report, snapshots, cuda, error

    def test_repeated_trials_restore_order_and_persist_partial_progress(self):
        report, snapshots, cuda, error = self.run_fake_experiment()
        self.assertIsNone(error)
        self.assertEqual(report["status"], "completed")
        self.assertEqual(report["provenance"], {"seed": 42})
        self.assertIsNone(snapshots[0]["reference"])
        self.assertTrue(any(snapshot["reference"] and len(snapshot["reference"].get("trials", [])) == 1
                            for snapshot in snapshots))
        self.assertEqual([(config["batch_size"], config["length_sorted"]) for config in report["optimized"]],
                         [(1, False), (8, False), (8, True), (16, True), (32, True)])
        for config in [report["reference"], *report["optimized"]]:
            self.assertEqual(config["status"], "completed")
            self.assertEqual(config["reference_parity"]["answer_agreement_rows"], 200)
            self.assertEqual(config["reference_parity"]["max_absolute_probability_difference"], 0)
            self.assertEqual(config["warmup_rows"], 220)
            self.assertEqual(len(config["trials"]), 2)
            for trial in config["trials"]:
                self.assertTrue(trial["minimum_duration_reached"])
                self.assertGreaterEqual(trial["measured_seconds"], 3)
                self.assertEqual([item["state"] for item in trial["results"]], list(map(str, range(20, 220))))
        cuda.empty_cache.assert_not_called()
        self.assertLess(report["optimized"][2]["padding_fraction"], report["optimized"][1]["padding_fraction"])

    def test_oom_reduces_cap_without_dropping_rows(self):
        report, snapshots, cuda, error = self.run_fake_experiment(oom_limit=16)
        self.assertIsNone(error)
        config = report["optimized"][-1]
        # The first 20-row warmup OOM is split into two safe batches of ten.
        self.assertEqual(config["runtime_batch_cap"], 10)
        self.assertTrue(config["warmup_stats"]["oom_splits"])
        self.assertFalse(config["trials"][0]["oom_splits"])
        self.assertEqual(config["reference_parity"]["answer_agreement_rows"], 200)
        self.assertTrue(cuda.empty_cache.called)

    def test_failed_config_keeps_committed_reference_and_failure_status(self):
        report, snapshots, cuda, error = self.run_fake_experiment(fail_optimized=True)
        self.assertIsNotNone(error)
        self.assertEqual(snapshots[-1]["status"], "failed")
        self.assertEqual(snapshots[-1]["reference"]["status"], "completed")
        self.assertEqual(len(snapshots[-1]["reference"]["trials"]), 2)
        self.assertEqual(snapshots[-1]["optimized"][0]["status"], "failed")
        cuda.empty_cache.assert_not_called()

    def test_duration_cap_is_explicit_and_invalid_scores_are_not_repaired(self):
        report, _, _, error = self.run_fake_experiment(min_seconds=60, max_passes=1, invalid=True)
        self.assertIsNone(error)
        config = report["optimized"][-1]
        self.assertFalse(config["trials"][0]["minimum_duration_reached"])
        self.assertEqual(config["results"][0]["probabilities"], ["nan", "inf"])
        self.assertEqual(config["reference_parity"]["compared_valid_rows"], 0)
        self.assertEqual(config["reference_parity"]["invalid_pair_indices"], list(range(200)))
        self.assertIsNone(config["reference_parity"]["answer_agreement_fraction"])
