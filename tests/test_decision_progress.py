import unittest
from unittest.mock import patch

from distill_jev.models.decision.perplexity import progress


class ProgressTests(unittest.TestCase):
    def test_coordinator_and_workers_share_run_clock(self):
        with (patch.object(progress, "time", return_value=145),
              patch.dict(progress.os.environ, {progress.RUN_STARTED_AT_ENV: "100"}),
              patch("builtins.print") as output):
            progress.log("worker one")
            progress.log("worker two")
            progress.log("coordinator", started_at=100)
        for call in output.call_args_list:
            self.assertIn("[run elapsed=45.0s]", call.args[0])

    def test_explicit_start_overrides_environment_for_new_run(self):
        with (patch.object(progress, "time", return_value=145),
              patch.dict(progress.os.environ, {progress.RUN_STARTED_AT_ENV: "100"}),
              patch("builtins.print") as output):
            progress.log("new run", started_at=140)
        self.assertIn("[run elapsed=5.0s]", output.call_args.args[0])
