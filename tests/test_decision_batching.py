"""CPU-only batching and OOM recovery checks."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from themis.models.decision.perplexity.batching import (
    plan_batches, predict_indexed_batch, predict_length_batched, rendered_input_lengths,
)


class DecisionBatchingTests(unittest.TestCase):
    def test_full_native_lengths_and_overflow(self):
        rows = [{"state": "a", "images": ["data:image/png;base64,test"]}]
        model = SimpleNamespace(prepare=MagicMock(return_value=SimpleNamespace(input_tokens=8192)))
        self.assertEqual(rendered_input_lengths(model, rows), [8192])
        model.prepare.assert_called_once_with(rows, max_length=8192)
        model.prepare.return_value = SimpleNamespace(input_tokens=8193)
        with self.assertRaises(ValueError):
            rendered_input_lengths(model, rows)

    def test_stable_sort_and_token_budget(self):
        self.assertEqual(plan_batches([3, 1, 2, 1], 2), [[1, 3], [2, 0]])
        self.assertEqual(plan_batches([8192, 1, 8192, 1, 5000], 32,
                                     max_padded_tokens=8192), [[1, 3], [4], [0], [2]])
        self.assertEqual([len(batch) for batch in plan_batches([100] * 65, 32)], [32, 32, 1])

    def test_sorted_prediction_preserves_scores_and_learns_oom_cap(self):
        rows = [{"state": index} for index in range(65)]
        calls = []

        def predict(batch, batch_size):
            calls.append(batch_size)
            if batch_size > 16:
                raise MemoryError("simulated CUDA OOM")
            return [[row["state"]] for row in batch]

        model = SimpleNamespace(
            prepare=lambda batch, max_length: SimpleNamespace(input_tokens=300 - batch[0]["state"]),
            predict=predict,
        )
        cuda = SimpleNamespace(OutOfMemoryError=MemoryError, synchronize=MagicMock(), empty_cache=MagicMock())
        with patch("themis.models.decision.perplexity.batching.import_module",
                   return_value=SimpleNamespace(cuda=cuda)):
            probabilities, stats = predict_length_batched(model, rows)
        self.assertEqual(probabilities, [[index] for index in range(65)])
        self.assertEqual(calls, [32, 16, 16, 16, 16, 1])
        self.assertEqual(stats["runtime_batch_cap"], 16)
        cuda.empty_cache.assert_called_once_with()

    def test_singleton_oom_other_errors_and_missing_scores_propagate(self):
        cuda = SimpleNamespace(OutOfMemoryError=MemoryError, synchronize=MagicMock(), empty_cache=MagicMock())
        model = SimpleNamespace(predict=MagicMock())
        stats = {"successful_batch_sizes": [], "oom_splits": []}
        for error in (MemoryError("singleton OOM"), RuntimeError("native failure")):
            model.predict.side_effect = error
            with self.assertRaises(type(error)):
                predict_indexed_batch(model, [{}], [0], SimpleNamespace(cuda=cuda), stats)
        model.predict.side_effect = None
        model.predict.return_value = []
        with self.assertRaises(ValueError):
            predict_indexed_batch(model, [{}], [0], SimpleNamespace(cuda=cuda), stats)

    def test_bad_batch_ceiling_and_overflow_fail_before_forward(self):
        model = SimpleNamespace(prepare=MagicMock(return_value=SimpleNamespace(input_tokens=8193)),
                                predict=MagicMock())
        for size in (0, 33):
            with self.assertRaises(ValueError):
                predict_length_batched(model, [{}], batch_size=size)
        model.prepare.assert_not_called()
        with self.assertRaises(ValueError):
            predict_length_batched(model, [{}])
        model.predict.assert_not_called()
