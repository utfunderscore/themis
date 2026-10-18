import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from themis.models.deberta import modal_training


class ModalTrainingTests(unittest.TestCase):
    def test_missing_split_fails_before_starting_modal(self):
        with TemporaryDirectory() as temp, patch.object(modal_training.app, "run") as run:
            with self.assertRaises(FileNotFoundError):
                modal_training.train(temp)
            run.assert_not_called()

    def test_uploads_splits_and_passes_config_to_remote_worker(self):
        with TemporaryDirectory() as temp:
            source = Path(temp)
            for split in ("train", "validation", "test"):
                (source / f"{split}.parquet").touch()
            with (
                patch.object(modal_training.app, "run"),
                patch.object(modal_training.modal, "enable_output"),
                patch.object(modal_training.volume, "batch_upload") as upload,
                patch.object(modal_training.train_remote, "remote", return_value={"metrics": {}}) as remote,
            ):
                result = modal_training.train(source, config={"bf16": True})
            run_id, config = remote.call_args.args
            self.assertEqual(config, {"bf16": True})
            self.assertEqual(result, {"metrics": {}})
            calls = upload.return_value.__enter__.return_value.put_file.call_args_list
            self.assertEqual(len(calls), 3)
            for call, split in zip(calls, ("train", "validation", "test"), strict=True):
                self.assertEqual(call.args, (
                    source / f"{split}.parquet", f"/runs/{run_id}/splits/{split}.parquet",
                ))

    def test_uploads_extra_splits_and_initial_model(self):
        with TemporaryDirectory() as temp:
            source = Path(temp) / "splits"
            source.mkdir()
            for split in ("train", "validation", "test", "teacher_test"):
                (source / f"{split}.parquet").touch()
            model = Path(temp) / "model"
            model.mkdir()
            (model / "config.json").touch()
            with (
                patch.object(modal_training.app, "run"),
                patch.object(modal_training.modal, "enable_output"),
                patch.object(modal_training.volume, "batch_upload") as upload,
                patch.object(modal_training.train_remote, "remote", return_value={}) as remote,
            ):
                modal_training.train(source, config={"model_name": "ignored"}, init_model_directory=model)
            run_id, config = remote.call_args.args
            self.assertEqual(config, {"model_name": f"/training/runs/{run_id}/init_model"})
            batch = upload.return_value.__enter__.return_value
            self.assertEqual(batch.put_file.call_args_list[-1].args, (
                source / "teacher_test.parquet", f"/runs/{run_id}/splits/teacher_test.parquet",
            ))
            batch.put_directory.assert_called_once_with(str(model), f"/runs/{run_id}/init_model")

    def test_missing_initial_model_fails_before_starting_modal(self):
        with TemporaryDirectory() as temp, patch.object(modal_training.app, "run") as run:
            for split in ("train", "validation", "test"):
                (Path(temp) / f"{split}.parquet").touch()
            with self.assertRaises(FileNotFoundError):
                modal_training.train(temp, init_model_directory=Path(temp) / "missing")
            run.assert_not_called()
