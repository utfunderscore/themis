"""Run the shared DeBERTa trainer on Modal without local training dependencies."""

from pathlib import Path
from uuid import uuid4

import modal


VOLUME_NAME = "themis-deberta-training"
ROOT = Path("/training")
app = modal.App("themis-deberta-training")
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.13")
    .pip_install(
        "polars>=2.0.0", "numpy>=2,<3", "torch==2.8.0",
        "transformers==4.57.1", "accelerate>=1.10,<2",
        "sentencepiece>=0.2,<0.3", "protobuf>=5,<8",
    )
    .env({"HF_HOME": "/training/huggingface"})
    .add_local_python_source("themis")
)


@app.function(
    image=image,
    gpu="H100",
    cpu=4,
    memory=32768,
    timeout=24 * 60 * 60,
    volumes={str(ROOT): volume},
    max_containers=1,
)
def train_remote(run_id: str, config: dict[str, object]) -> dict[str, object]:
    """Train uploaded splits and commit artifacts before returning their locations."""
    import torch

    from .training import TrainingConfig, train

    if len(run_id) != 32 or any(char not in "0123456789abcdef" for char in run_id):
        raise ValueError("run_id must be a lowercase UUID hex string")
    settings = TrainingConfig(**config)
    if settings.use_cpu:
        raise ValueError("Modal training requires use_cpu=False")
    if not torch.cuda.is_available():
        raise RuntimeError("Modal training requires an available CUDA GPU")
    volume.reload()
    run_directory = ROOT / "runs" / run_id
    try:
        result = train(run_directory / "splits", run_directory / "output", config=settings)
    finally:
        # Preserve partial checkpoints as well as successful runs.
        volume.commit()
    return {
        "run_id": run_id,
        "volume_name": VOLUME_NAME,
        "output_directory": f"runs/{run_id}/output",
        "model_directory": f"runs/{run_id}/output/model",
        "metrics": result["metrics"],
    }


def train(
    split_directory: str | Path,
    *,
    config: dict[str, object] | None = None,
) -> dict[str, object]:
    """Upload prepared splits and wait for GPU training; return Volume paths/metrics.

    Config accepts TrainingConfig fields as a plain dictionary, so the caller
    does not need PyTorch or Transformers installed. Every call creates a new run.
    """
    source = Path(split_directory)
    paths = [source / f"{split}.parquet" for split in ("train", "validation", "test")]
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    settings = dict(config or {})
    if settings.get("use_cpu"):
        raise ValueError("Modal training requires use_cpu=False")
    run_id = uuid4().hex
    with modal.enable_output(), app.run():
        with volume.batch_upload() as upload:
            for path in paths:
                upload.put_file(path, f"/runs/{run_id}/splits/{path.name}")
        print(f"DeBERTa run: {VOLUME_NAME}/runs/{run_id}", flush=True)
        return train_remote.remote(run_id, settings)
