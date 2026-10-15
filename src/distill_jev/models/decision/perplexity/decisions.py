"""Pinned BF16 Perplexity inference for training-data generation on Modal."""

import json
import sys
from importlib import import_module
from pathlib import Path
from typing import Any

import modal

from .batching import DEFAULT_BATCH_SIZE, plan_batches, predict_length_batched
from .policy import INFERENCE_PACKAGES, MODEL_ID, MODEL_REVISION
from .progress import log, stage

MODEL_DIR = f"/models/{MODEL_REVISION}"
app = modal.App("distill-jev-decisions")
volume = modal.Volume.from_name("distill-jev-decision-models", create_if_missing=True)
data_volume = modal.Volume.from_name("distill-jev-decision-data", create_if_missing=True)
completed = modal.Dict.from_name("distill-jev-decision-shards", create_if_missing=True)
download_image = modal.Image.debian_slim(python_version="3.13").pip_install(
    "huggingface-hub==1.31.0"
)
# Build against the pinned Torch ABI; retain the optimized inference path.
inference_image = (
    modal.Image.from_registry("nvidia/cuda:13.0.2-devel-ubuntu24.04", add_python="3.13")
    .apt_install("build-essential")
    .pip_install("huggingface-hub==1.31.0", *INFERENCE_PACKAGES,
                 "setuptools", "wheel", "packaging", "ninja")
    .env({"CAUSAL_CONV1D_FORCE_BUILD": "TRUE", "MAX_JOBS": "2",
          "CC": "gcc", "CXX": "g++"})
    .run_commands("python -m pip install --no-build-isolation --no-deps causal-conv1d==1.7.0",
                  "python -c 'import causal_conv1d, causal_conv1d_cuda; print(causal_conv1d.__version__)'")
    .env({"HF_HUB_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false"})
)


def _ensure_checkpoint() -> str:
    """Reuse complete pinned weights; interrupted downloads are retried."""
    volume.reload()
    checkpoint = Path(MODEL_DIR)
    marker = checkpoint / ".download-complete.json"
    if marker.exists():
        saved = json.loads(marker.read_text())
        if saved.get("revision") == MODEL_REVISION and saved.get("files") and all(
            (checkpoint / name).is_file() and (checkpoint / name).stat().st_size == size
            for name, size in saved["files"].items()
        ):
            log("Checkpoint cache verified")
            return MODEL_DIR
    hub = import_module("huggingface_hub")
    with stage(f"Download/cache checkpoint {MODEL_REVISION}"):
        hub.snapshot_download(
            MODEL_ID, revision=MODEL_REVISION, local_dir=MODEL_DIR,
            allow_patterns=[
                "*.json", "*.jinja", "*.safetensors", "requirements.txt",
                "LICENSE", "NOTICE", "source/src/autojev/*.py",
            ],
        )
    index = json.loads((checkpoint / "model.safetensors.index.json").read_text())
    required = set(index["weight_map"].values()) | {
        "model.safetensors.index.json", "readout.safetensors", "decision_config.json",
        "config.json", "tokenizer.json", "tokenizer_config.json", "processor_config.json",
        "chat_template.jinja", "source/src/autojev/model.py", "source/src/autojev/types.py",
        "source/src/autojev/__init__.py",
    }
    sizes = {name: (checkpoint / name).stat().st_size for name in sorted(required)}
    volume.commit()
    marker.write_text(json.dumps({"revision": MODEL_REVISION, "files": sizes}))
    volume.commit()
    return MODEL_DIR


@app.function(image=download_image, volumes={"/models": volume}, timeout=3600)
def download_model() -> str:
    return _ensure_checkpoint()


def load_native_model() -> tuple[Any, Any]:
    volume.reload()
    checkpoint = Path(MODEL_DIR)
    sys.path.insert(0, str(checkpoint / "source" / "src"))
    with stage("Load native BF16 model"):
        native = import_module("autojev.model")
        model = native.DecisionModel(checkpoint, device="cuda", max_image_pixels=512 * 512)
    return model, native.answer


@app.cls(
    image=inference_image, gpu="H100!", cpu=8, memory=131072,
    volumes={"/models": volume, "/data": data_volume}, max_containers=1,
    scaledown_window=60, timeout=1800,
)
class Decisions:
    model: Any = None
    answer: Any = None

    @modal.enter()
    def load(self) -> None:
        self.model, self.answer = load_native_model()

    @modal.method()
    def predict(self, rows: list[dict[str, Any]],
                batch_size: int = DEFAULT_BATCH_SIZE) -> list[dict[str, Any]]:
        probabilities, stats = predict_length_batched(self.model, rows, batch_size)
        log(f"Predicted {len(rows)} rows; batch cap={stats['runtime_batch_cap']}, "
            f"OOM splits={len(stats['oom_splits'])}")
        return [
            {"probabilities": scores, "answer": self.answer(row["question"], scores),
             "model_id": MODEL_ID, "model_revision": MODEL_REVISION}
            for row, scores in zip(rows, probabilities, strict=True)
        ]

    @modal.method()
    def predict_file(self, input_path: str, batch_size: int) -> dict[str, Any]:
        from .bulk import _predict_file

        return _predict_file(input_path, batch_size, self.predict.local)


def predict_decisions(rows: list[dict[str, Any]], *,
                      batch_size: int = DEFAULT_BATCH_SIZE) -> list[dict[str, Any]]:
    """Score 1–128 native decision rows in a paid Modal run.

    Callers supply native state/question objects and optional data-URL images.
    Native inference handles their schema. Length sorting, token-budget limits,
    and OOM splitting preserve input order without truncating full inputs.
    """
    if not 1 <= len(rows) <= 128:
        raise ValueError("Provide between 1 and 128 decisions per call")
    plan_batches([], batch_size)
    with modal.enable_output(), app.run():
        download_model.remote()
        return Decisions().predict.remote(rows, batch_size=batch_size)
