"""Cached training, artifact download, and ONNX export for the DeBERTa student."""

import json
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory

from .dataset import prepare_dataset as prepare_training_dataset


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
    """Reuse or train a recorded Modal run and return its downloaded model path."""
    splits = Path(training_data_directory)
    # Keep the existing default receipt; custom split directories own their receipts.
    receipt = (splits.parent / "review-modal-run.json"
               if splits == Path("cache/deberta/review-splits")
               else splits / "modal-run.json")
    if receipt.is_file():
        print(f"Reusing completed Modal training: {receipt}", flush=True)
        return download_student(result=json.loads(receipt.read_text(encoding="utf-8")))

    from .modal_training import train

    if all((splits / f"{split}.parquet").is_file() for split in ("train", "validation", "test")):
        print(f"Reusing training splits: {splits}", flush=True)
    else:
        prepare_training_dataset(source_path=source, output_directory=splits)
    result = train(split_directory=splits)
    temporary = receipt.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    temporary.replace(receipt)
    return download_student(result=result)


def download_student(result: dict, cache_directory: str | Path = "cache/deberta/runs") -> Path:
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
