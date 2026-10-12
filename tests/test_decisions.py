"""Local contract checks; these never invoke Modal."""

import unittest
import json
import random
from pathlib import Path
from tempfile import TemporaryDirectory
from contextlib import nullcontext
from functools import wraps
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from distill_jev.models.decision.perplexity.decisions import validate_rows
from distill_jev.models.decision.perplexity.benchmark import sample_benchmark_rows
from distill_jev.models.decision.perplexity.experiment import (
    benchmark_loaded_model, compare_predictions, convolution_backends, validate_convolution_cuda,
)


class DecisionValidationTests(unittest.TestCase):
    def test_public_default_is_optimized_batch_thirty_two(self):
        from distill_jev.models.decision.perplexity import decisions

        rows = [{"state": "short message", "question": {"type": "noul"}}]
        with (patch.object(decisions.modal, "enable_output", return_value=nullcontext()),
              patch.object(decisions, "optimization_app") as app,
              patch.object(decisions, "download_experiment_model") as download,
              patch.object(decisions, "OptimizationExperiment") as worker,
              patch.object(decisions, "app") as baseline_app,
              patch.object(decisions, "Decisions") as baseline_worker):
            app.run.return_value = nullcontext()
            worker.return_value.predict.remote.return_value = [{"answer": True}]
            self.assertEqual(decisions.predict_decisions(rows), [{"answer": True}])
            worker.return_value.predict.remote.assert_called_once_with(rows, batch_size=32)
            download.remote.assert_called_once_with()
            baseline_app.run.assert_not_called()
            baseline_worker.assert_not_called()
            worker.return_value.predict.remote.reset_mock()
            decisions.predict_decisions(rows, batch_size=1)
            worker.return_value.predict.remote.assert_called_once_with(rows, batch_size=1)

    def test_public_batch_ceiling_validated_before_remote_run(self):
        from distill_jev.models.decision.perplexity import decisions

        rows = [{"state": "example", "question": {"type": "noul"}}]
        validate_rows(rows, 32)
        with patch.object(decisions, "optimization_app") as app:
            for size in (0, 33, True, 1.5):
                with self.assertRaises(ValueError):
                    decisions.predict_decisions(rows, batch_size=size)
            app.run.assert_not_called()
        with self.assertRaises(ValueError):
            validate_rows(rows, 32, max_batch_size=8)

    def test_wrapper_identity_is_not_kernel_availability(self):
        def reference():
            pass
        reference.__module__ = "fake_qwen"
        @wraps(reference)
        def wrapper():
            return None
        optimized = lambda: None
        qwen = SimpleNamespace(__name__="fake_qwen", causal_conv1d_fn=wrapper)
        self.assertEqual(convolution_backends(qwen, SimpleNamespace(
            causal_conv1d_fn=optimized)), (reference, optimized))

    def test_tiny_cuda_validation_requires_real_native_dispatch(self):
        tensor = MagicMock()
        tensor.dtype, tensor.device = "BF16", "cuda"
        tensor.shape = (1, 64, 32)
        tensor.stride.return_value = (2048, 1, 64)
        tensor.transpose.return_value = tensor
        torch = MagicMock()
        torch.testing = SimpleNamespace(assert_close=MagicMock())
        torch.__version__ = "test"
        torch.bfloat16 = "BF16"
        torch.randn.return_value = tensor
        torch.nn.functional.silu.return_value = tensor
        torch.cuda.is_available.return_value = True
        interface = SimpleNamespace(causal_conv1d_fwd_function=lambda *a, **kw: tensor)
        def optimized(*args, **kwargs):
            return interface.causal_conv1d_fwd_function(*args, **kwargs)
        qwen = SimpleNamespace(causal_conv1d_fn=optimized)
        modules = {"torch": torch, "causal_conv1d": SimpleNamespace(causal_conv1d_fn=optimized),
                   "causal_conv1d.causal_conv1d_interface": interface,
                   "transformers.models.qwen3_5.modeling_qwen3_5": qwen}
        original = interface.causal_conv1d_fwd_function
        target = "distill_jev.models.decision.perplexity.experiment"
        with patch(f"{target}.import_module", side_effect=modules.__getitem__):
            report = validate_convolution_cuda()
            self.assertEqual(len(report["cases"]), 4)
            self.assertFalse(report["model_weights_loaded"])
            self.assertTrue(all(case["native_cuda_forward_calls"] == 1 for case in report["cases"]))
            self.assertIs(interface.causal_conv1d_fwd_function, original)
            qwen.causal_conv1d_fn = lambda *args, **kwargs: tensor
            with self.assertRaisesRegex(RuntimeError, "Package CUDA test passed"):
                validate_convolution_cuda()
            self.assertIs(interface.causal_conv1d_fwd_function, original)
            qwen.causal_conv1d_fn = optimized
            torch.testing.assert_close.side_effect = AssertionError("numerical mismatch")
            with self.assertRaisesRegex(AssertionError, "numerical mismatch"):
                validate_convolution_cuda()
            torch.cuda.is_available.return_value = False
            with self.assertRaisesRegex(RuntimeError, "requires CUDA"):
                validate_convolution_cuda()

    def test_complete_checkpoint_skips_hub_and_commit(self):
        from distill_jev.models.decision.perplexity import decisions

        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "weight").write_bytes(b"weights")
            (root / ".download-complete.json").write_text(json.dumps({
                "revision": decisions.MODEL_REVISION, "files": {"weight": 7},
            }))
            with (patch.object(decisions, "MODEL_DIR", directory),
                  patch.object(decisions, "volume") as volume,
                  patch.object(decisions, "import_module") as imports):
                self.assertEqual(decisions._ensure_checkpoint(), directory)
                imports.assert_not_called()
                volume.commit.assert_not_called()
                # An incomplete/truncated cache must go through the downloader.
                (root / "weight").write_bytes(b"bad")
                imports.side_effect = RuntimeError("downloader requested")
                with self.assertRaisesRegex(RuntimeError, "downloader requested"):
                    decisions._ensure_checkpoint()

    def test_sampling_preserves_baseline_seed_and_order(self):
        rng = random.Random(42)
        expected = list(range(220))
        for index in range(220, 300):
            selected = rng.randrange(index + 1)
            if selected < 220:
                expected[selected] = index
        with TemporaryDirectory() as directory:
            source = Path(directory) / "sample.jsonl"
            source.write_text("".join(json.dumps({"content": str(index)}) + "\n"
                                      for index in range(300)))
            rows, indices, total = sample_benchmark_rows(source)
        self.assertEqual(total, 300)
        self.assertEqual(indices, expected)
        self.assertEqual([row["state"] for row in rows], list(map(str, expected)))

    def test_native_question_types(self):
        for question in (
            {"type": "choice", "criteria": {"a": "A", "b": "B"}},
            {"type": "score", "criteria": ["low", "high"]},
            {"type": "noul"},
        ):
            validate_rows([{"state": "example", "question": question}], 1)

    def test_invalid_requests_rejected_before_remote_run(self):
        for question in (
            {"type": "choice", "criteria": {}},
            {"type": "score", "criteria": ["only"]},
            {"type": "unknown"},
            {"type": "noul", "criteria": {"yes": "Yes"}},
        ):
            with self.assertRaises(ValueError):
                validate_rows([{"state": "example", "question": question}], 1)

    def test_local_image_paths_rejected(self):
        with self.assertRaises(ValueError):
            validate_rows([{
                "state": "example", "question": {"type": "noul"},
                "images": ["/local/image.png"],
            }], 1)

    def test_nan_rejected(self):
        with self.assertRaises(ValueError):
            validate_rows([{"state": float("nan"), "question": {"type": "noul"}}], 1)

    def test_prediction_comparison(self):
        reference = [{"probabilities": [0.75, 0.25], "answer": "a"}]
        candidate = [{"probabilities": [0.5, 0.5], "answer": "b"}]
        result = compare_predictions(reference, candidate)
        self.assertEqual(result["changed_answer_indices"], [0])
        self.assertEqual(result["max_absolute_probability_difference"], 0.25)
        self.assertEqual(result["answer_agreement_fraction"], 0)
        with self.assertRaises(ValueError):
            compare_predictions(reference, [])
        result = compare_predictions(
            [{"probabilities": [0.75, 0.25], "answer": {
                "type": "choice", "choice": "a", "confidence": 0.5}}],
            [{"probabilities": [0.74, 0.26], "answer": {
                "type": "choice", "choice": "a", "confidence": 0.48}}],
        )
        self.assertEqual(result["answer_agreement_rows"], 1)
        self.assertEqual(result["exact_native_answer_agreement_rows"], 0)
        result = compare_predictions(reference, [{"error": "invalid", "answer": None}])
        self.assertEqual(result["compared_valid_rows"], 0)
        self.assertEqual(result["invalid_pair_indices"], [0])
        self.assertIsNone(result["answer_agreement_fraction"])

    def test_experiment_warmup_batches_and_scoped_reference(self):
        def reference_function():
            pass

        reference_function.__module__ = "fake_qwen"

        def optimized_function():
            pass

        def wrap(implementation):
            @wraps(reference_function)
            def selected():
                return implementation()
            return selected

        selected = wrap(optimized_function)
        def module_wrapper(func):
            class Func:
                def forward(self, *args, **kwargs):
                    return func(*args, **kwargs)
            return Func()
        selected = module_wrapper(selected)
        qwen = SimpleNamespace(__name__="fake_qwen", causal_conv1d_fn=selected)
        cuda = SimpleNamespace(
            synchronize=lambda: None, empty_cache=lambda: None,
            reset_peak_memory_stats=lambda: None,
            max_memory_allocated=lambda: 1024, max_memory_reserved=lambda: 2048,
            get_device_name=lambda _: "fake GPU",
            get_device_properties=lambda _: SimpleNamespace(total_memory=4096),
            OutOfMemoryError=MemoryError,
        )
        torch = SimpleNamespace(cuda=cuda, bfloat16="BF16", version=SimpleNamespace(cuda="13.0"))
        modules = {
            "torch": torch,
            "transformers.models.qwen3_5.modeling_qwen3_5": qwen,
            "causal_conv1d": SimpleNamespace(causal_conv1d_fn=optimized_function),
        }
        calls = []

        def predict(rows, batch_size):
            calls.append((len(rows), batch_size, qwen.causal_conv1d_fn is reference_function))
            return [[0.75, 0.25] for _ in rows]

        model = SimpleNamespace(
            predict=predict,
            backbone=SimpleNamespace(parameters=lambda: iter([SimpleNamespace(dtype="BF16")])),
        )
        target = "distill_jev.models.decision.perplexity.experiment"
        with (patch(f"{target}.import_module", side_effect=modules.__getitem__),
              patch(f"{target}.metadata.version", return_value="test"),
              patch(f"{target}.log"), patch(f"{target}.log_count"),
              patch(f"{target}.stage", side_effect=lambda _: nullcontext())):
            report = benchmark_loaded_model(
                model, lambda question, scores: "a",
                [{"state": "example", "question": {"type": "noul"}}] * 220,
                1.0, model_id="test", model_revision="revision",
            )
            with patch.object(model, "predict", side_effect=RuntimeError("test failure")):
                with self.assertRaisesRegex(RuntimeError, "test failure"):
                    benchmark_loaded_model(
                        model, lambda question, scores: "a",
                        [{"state": "example", "question": {"type": "noul"}}] * 220,
                        1.0, model_id="test", model_revision="revision",
                    )
            def reject_invalid(question, scores):
                raise ValueError("nonfinite")
            with patch.object(model, "predict", side_effect=lambda rows, batch_size:
                              [[float("nan"), float("inf")] for _ in rows]):
                invalid_report = benchmark_loaded_model(
                    model, reject_invalid,
                    [{"state": "example", "question": {"type": "noul"}}] * 220,
                    1.0, model_id="test", model_revision="revision",
                )
        self.assertIs(qwen.causal_conv1d_fn, selected)
        self.assertEqual(json.loads(json.dumps(report, allow_nan=False))["dtype"], "bfloat16")
        json.dumps(invalid_report, allow_nan=False)
        self.assertEqual(len(invalid_report["reference"]["invalid_prediction_indices"]), 200)
        self.assertEqual(invalid_report["reference"]["results"][0]["probabilities"], ["nan", "inf"])
        self.assertEqual(invalid_report["optimized"][0]["reference_parity"]["compared_valid_rows"], 0)
        self.assertEqual(len(calls), 633)
        self.assertEqual(sum(is_reference for _, _, is_reference in calls), 220)
        self.assertEqual([run["warmup_calls"] for run in report["optimized"]], [20, 10, 5, 3])
        self.assertEqual(calls[-26], (4, 8, False))
        for run in report["optimized"]:
            self.assertEqual(len(run["results"]), 200)
            self.assertEqual(run["reference_parity"]["answer_agreement_rows"], 200)
