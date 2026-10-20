"""Cached training, artifact download, and ONNX export for the DeBERTa student."""

import json
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory
from typing import Any

from . import BASE_MODEL
from .dataset import prepare_dataset as prepare_training_dataset
from .report import format_comparison, format_run_metrics

# Receipts written before the base model was recorded are DeBERTa-v3 runs.
_UNRECORDED_BASE_MODEL = "microsoft/deberta-v3-base"


def train_and_export(
    source: str | Path,
    *,
    training_data_directory: str | Path = "cache/deberta/review-splits",
    onnx_output_directory: str | Path | None = None,
) -> Path:
    """Reuse completed training/export steps and return the ONNX model.

    Training splits are written to training_data_directory. When supplied,
    onnx_output_directory contains one subdirectory per training run ID;
    otherwise exports remain alongside the downloaded checkpoint.
    """
    model_directory = train_student(source=source, training_data_directory=training_data_directory)
    return export_if_missing(model_directory, onnx_output_directory)


def correct_and_export(
    labeled_path: str | Path,
    model_directory: str | Path,
    *,
    teacher_split_directory: str | Path = "cache/deberta/review-splits",
    correction_data_directory: str | Path = "cache/deberta/correction-splits",
    onnx_output_directory: str | Path | None = None,
) -> Path:
    """Post-train an already trained student (see correct_student); return its ONNX model.

    The corrected model has its own run ID, so its export never replaces the
    distilled model's.
    """
    corrected = correct_student(
        labeled_path, model_directory, teacher_split_directory=teacher_split_directory,
        correction_data_directory=correction_data_directory,
    )
    return export_if_missing(corrected, onnx_output_directory)


def export_if_missing(model_directory: Path, onnx_output_directory: str | Path | None = None) -> Path:
    """Export model_directory to ONNX unless that export already exists; return model.onnx."""
    onnx_directory = (model_directory.parent / "onnx" if onnx_output_directory is None
                      else Path(onnx_output_directory) / model_directory.parent.name)
    if (onnx_directory / "model.onnx").is_file() and (onnx_directory / "inference_config.json").is_file():
        print(f"Reusing completed ONNX export: {onnx_directory}", flush=True)
        return onnx_directory / "model.onnx"
    return export_student(model_directory=model_directory, output_directory=onnx_directory)


def export_student(
    model_directory: str | Path,
    output_directory: str | Path = "cache/deberta/onnx",
    *,
    max_length: int | None = None,
) -> Path:
    """Export a local/downloaded training run; requires training and onnx extras."""
    from .onnx_export import export_onnx

    return export_onnx(model_directory=model_directory, output_directory=output_directory, max_length=max_length)


def train_student(
    source: str | Path,
    *,
    training_data_directory: str | Path = "cache/deberta/review-splits",
) -> Path:
    """Reuse or train a recorded Modal run and return its downloaded model path.

    A recorded run is reused only if it was trained from BASE_MODEL. Otherwise
    a new run is trained and the old receipt is kept, suffixed with its run ID.
    """
    splits = Path(training_data_directory)
    # Keep the existing default receipt; custom split directories own their receipts.
    receipt = (splits.parent / "review-modal-run.json"
               if splits == Path("cache/deberta/review-splits")
               else splits / "modal-run.json")
    superseded = None
    if receipt.is_file():
        previous = json.loads(receipt.read_text(encoding="utf-8"))
        trained = previous.get("model_name", _UNRECORDED_BASE_MODEL)
        if trained == BASE_MODEL:
            print(f"Reusing completed Modal training: {receipt}", flush=True)
            _print_run_metrics("Training metrics", previous.get("metrics"))
            return download_student(result=previous)
        superseded = receipt.with_name(f"{receipt.stem}-{previous['run_id']}{receipt.suffix}")
        print(f"Recorded run {receipt} used {trained}; training {BASE_MODEL}", flush=True)

    from .modal_training import train

    if all((splits / f"{split}.parquet").is_file() for split in ("train", "validation", "test")):
        print(f"Reusing training splits: {splits}", flush=True)
    else:
        prepare_training_dataset(source_path=source, output_directory=splits)
    result = train(split_directory=splits)
    if superseded is not None:
        receipt.replace(superseded)
    temporary = receipt.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    temporary.replace(receipt)
    _print_run_metrics("Training metrics", result.get("metrics"))
    return download_student(result=result)


# Continue gently from the distilled weights rather than retraining from scratch.
CORRECTION_TRAINING = {"epochs": 2.0, "learning_rate": 1e-5, "warmup_ratio": 0.06}


def correct_student(
    labeled_path: str | Path,
    model_directory: str | Path,
    *,
    teacher_split_directory: str | Path = "cache/deberta/review-splits",
    correction_data_directory: str | Path = "cache/deberta/correction-splits",
    config: Mapping[str, object] | None = None,
) -> Path:
    """Post-train a distilled student on reviewed labels; return the new model path.

    labeled_path is reviewed JSONL (text, category), such as
    moderation_audit/all_samples.jsonl. model_directory is the local student to
    continue from. config overrides CORRECTION_TRAINING. Splits and the run
    receipt live in correction_data_directory; a recorded run is reused only if
    it continued from the same model_directory. Otherwise the old receipt is
    kept, suffixed with its run ID.
    """
    from .correction import prepare_correction_dataset

    splits = Path(correction_data_directory)
    receipt = splits / "modal-run.json"
    init_model = str(Path(model_directory).resolve())
    superseded = None
    if receipt.is_file():
        previous = json.loads(receipt.read_text(encoding="utf-8"))
        if previous.get("init_model_directory") == init_model:
            print(f"Reusing completed correction training: {receipt}", flush=True)
            _print_run_metrics("Post-training metrics", previous.get("metrics"))
            return download_student(result=previous)
        superseded = receipt.with_name(f"{receipt.stem}-{previous['run_id']}{receipt.suffix}")
        print(f"Recorded correction run {receipt} started from another model; retraining", flush=True)

    from .modal_training import train

    if (splits / "manifest.json").is_file():
        print(f"Reusing correction splits: {splits}", flush=True)
    else:
        prepare_correction_dataset(labeled_path, teacher_split_directory, splits)
    result = train(split_directory=splits, config={**CORRECTION_TRAINING, **(config or {})},
                   init_model_directory=model_directory)
    result = {**result, "init_model_directory": init_model}
    if superseded is not None:
        receipt.replace(superseded)
    temporary = receipt.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    temporary.replace(receipt)
    _print_run_metrics("Post-training metrics", result.get("metrics"))
    return download_student(result=result)


def train_student_locally(
    source: str | Path,
    *,
    training_data_directory: str | Path = "cache/deberta/review-splits",
    output_directory: str | Path = "cache/deberta/local-run",
    config: Mapping[str, object] | None = None,
) -> Path:
    """Train BASE_MODEL from scratch on this machine; return the model path.

    Requires the training extra. A completed run in output_directory is
    reused only if it was trained from BASE_MODEL; a run from another base
    model, or an incomplete run, must be moved aside first.
    """
    output = Path(output_directory)
    if _completed_local_run(output, BASE_MODEL):
        print(f"Reusing completed local training: {output}", flush=True)
    else:
        from .training import TrainingConfig, train

        splits = Path(training_data_directory)
        if all((splits / f"{split}.parquet").is_file() for split in ("train", "validation", "test")):
            print(f"Reusing training splits: {splits}", flush=True)
        else:
            prepare_training_dataset(source_path=source, output_directory=splits)
        train(splits, output, config=TrainingConfig.from_mapping({**(config or {}), "model_name": BASE_MODEL}))
    _print_run_metrics("Training metrics", _read_json(output / "metrics.json"))
    return output / "model"


def correct_student_locally(
    labeled_path: str | Path,
    model_directory: str | Path,
    *,
    teacher_split_directory: str | Path = "cache/deberta/review-splits",
    correction_data_directory: str | Path = "cache/deberta/correction-splits",
    output_directory: str | Path = "cache/deberta/local-correction-run",
    config: Mapping[str, object] | None = None,
) -> Path:
    """Post-train a trained student on this machine (see correct_student); return the model path.

    A completed run in output_directory is reused only if it continued from
    model_directory; otherwise it must be moved aside first.
    """
    output, init_model = Path(output_directory), str(Path(model_directory).resolve())
    if _completed_local_run(output, init_model):
        print(f"Reusing completed local correction training: {output}", flush=True)
    else:
        from .correction import prepare_correction_dataset
        from .training import TrainingConfig, train

        splits = Path(correction_data_directory)
        if (splits / "manifest.json").is_file():
            print(f"Reusing correction splits: {splits}", flush=True)
        else:
            prepare_correction_dataset(labeled_path, teacher_split_directory, splits)
        settings = TrainingConfig.from_mapping({**CORRECTION_TRAINING, **(config or {}), "model_name": init_model})
        train(splits, output, config=settings)
    _print_run_metrics("Post-training metrics", _read_json(output / "metrics.json"))
    return output / "model"


def compare_correction_locally(
    model_directory: str | Path,
    corrected_directory: str | Path,
    *,
    correction_data_directory: str | Path = "cache/deberta/correction-splits",
    config: Mapping[str, object] | None = None,
) -> dict[str, dict[str, dict[str, float]]]:
    """Compare the distilled and post-trained students on the correction test splits.

    corrected_directory is the model returned by correct_student_locally; its
    run's metrics.json supplies the post-trained side. model_directory is
    evaluated on the same test and teacher_test splits, and those metrics are
    cached in comparison.json beside the corrected run, reused while they came
    from the same model_directory. config accepts TrainingConfig evaluation
    fields. Returns {"distilled": ..., "post_trained": ...} keyed by split.
    """
    run = Path(corrected_directory).parent
    cache = run / "comparison.json"
    baseline = str(Path(model_directory).resolve())
    splits = ("test", "teacher_test")
    post_trained = _read_json(run / "metrics.json")
    previous = _read_json(cache) if cache.is_file() else {}
    if previous.get("distilled_model") == baseline:
        print(f"Reusing completed comparison: {cache}", flush=True)
        distilled = previous["distilled"]
    else:
        from .training import TrainingConfig, evaluate

        directory = Path(correction_data_directory)
        settings = TrainingConfig.from_mapping({**(config or {}), "model_name": baseline})
        distilled = evaluate(model_directory, {split: directory / f"{split}.parquet" for split in splits},
                             config=settings)
        temporary = cache.with_suffix(".json.tmp")
        temporary.write_text(json.dumps({"distilled_model": baseline, "distilled": distilled}, indent=2) + "\n",
                             encoding="utf-8")
        temporary.replace(cache)
    comparison = {"distilled": distilled, "post_trained": {split: post_trained[split] for split in splits}}
    print("Distilled vs post-trained (test = held-out reviewed labels, teacher_test = teacher labels)",
          flush=True)
    print(format_comparison(comparison["distilled"], comparison["post_trained"]), flush=True)
    return comparison


def _print_run_metrics(title: str, metrics: object) -> None:
    if isinstance(metrics, Mapping) and metrics:
        print(f"{title}\n{format_run_metrics(metrics)}", flush=True)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _completed_local_run(output: Path, model_name: str) -> bool:
    """Whether output holds a finished run from model_name; raise if it holds anything else."""
    if not output.exists():
        return False
    run_config = output / "run_config.json"
    trained = (json.loads(run_config.read_text(encoding="utf-8")).get("model_name")
               if run_config.is_file() else None)
    if trained == model_name and (output / "metrics.json").is_file() and (output / "model" / "config.json").is_file():
        return True
    raise FileExistsError(f"{output} holds an incomplete run or one from {trained}; move it aside to retrain")


def download_student(result: Mapping[str, Any], cache_directory: str | Path = "cache/deberta/runs") -> Path:
    """Atomically cache model/tokenizer and run config from the exact training receipt."""
    run_id = result["run_id"]
    if not isinstance(run_id, str) or len(run_id) != 32 or any(c not in "0123456789abcdef" for c in run_id):
        raise ValueError("Invalid training run ID")
    identity = {key: result[key] for key in ("run_id", "volume_name", "output_directory", "model_directory")}
    destination = Path(cache_directory) / run_id
    marker = destination / "training_source.json"
    if destination.exists():
        if (marker.is_file() and json.loads(marker.read_text(encoding="utf-8")) == identity
                and (destination / "model" / "config.json").is_file()
                and (destination / "run_config.json").is_file()):
            return destination / "model"
        raise ValueError(f"Incomplete or mismatched training artifacts at {destination}")

    import modal

    volume = modal.Volume.from_name(result["volume_name"])
    remote_model = PurePosixPath(result["model_directory"].lstrip("/"))
    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="download-student-", dir=destination.parent) as temporary:
        bundle = Path(temporary) / "bundle"
        (bundle / "model").mkdir(parents=True)
        for entry in volume.iterdir(str(remote_model), recursive=True):
            if entry.type.name != "FILE":
                continue
            relative = PurePosixPath(entry.path.lstrip("/")).relative_to(remote_model)
            if ".." in relative.parts:
                raise ValueError("Invalid model artifact path")
            local = bundle / "model" / str(relative)
            local.parent.mkdir(parents=True, exist_ok=True)
            with local.open("wb") as writer:
                for chunk in volume.read_file(entry.path):
                    writer.write(chunk)
        with (bundle / "run_config.json").open("wb") as writer:
            for chunk in volume.read_file(str(PurePosixPath(result["output_directory"]) / "run_config.json")):
                writer.write(chunk)
        if not (bundle / "model" / "config.json").is_file():
            raise FileNotFoundError("Downloaded model is missing config.json")
        (bundle / "training_source.json").write_text(json.dumps(identity, indent=2) + "\n", encoding="utf-8")
        bundle.rename(destination)
    return destination / "model"
