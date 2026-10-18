import unittest

from themis.models.deberta.report import format_comparison, format_run_metrics


class MetricsReportTests(unittest.TestCase):
    def test_run_metrics_have_one_column_per_split_without_throughput(self):
        table = format_run_metrics({
            "train": {"train_loss": 0.5, "epoch": 3.0, "train_runtime": 9.0},
            "validation": {"validation_loss": 0.25, "validation_runtime": 1.0, "epoch": 3.0},
            "test": {"test_loss": 0.125, "test_category_teacher_agreement": 0.9},
            "best_checkpoint": "run/checkpoints/checkpoint-2",
        })
        lines = table.splitlines()
        self.assertEqual(lines[0].split(), ["metric", "validation", "test"])
        self.assertEqual(lines[2].split(), ["loss", "0.2500", "0.1250"])
        self.assertEqual(lines[3].split(), ["category_teacher_agreement", "-", "0.9000"])
        self.assertNotIn("runtime", table)
        self.assertIn("train loss 0.5000 over 3 epochs", table)
        self.assertIn("checkpoint-2", table)

    def test_comparison_shows_change_for_shared_splits(self):
        table = format_comparison(
            {"test": {"test_loss": 0.5, "test_runtime": 1.0}, "teacher_test": {"teacher_test_loss": 0.25}},
            {"test": {"test_loss": 0.375}, "validation": {"validation_loss": 0.1}},
        )
        self.assertTrue(table.startswith("test\n"))
        self.assertIn("-0.1250", table)
        self.assertNotIn("teacher_test", table)
        self.assertNotIn("runtime", table)
