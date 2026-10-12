"""Pinned Perplexity decision inference on Modal. Importing does not run Modal.

Call predict_decisions explicitly to start a paid remote run. The upstream
predictor owns prompting, padding, noncausal attention, pooling and calibration.
"""

import json
import sys
from importlib import import_module
from pathlib import Path
from time import perf_counter
from typing import Any

import modal

from .progress import log, log_count, stage
from .batching import DEFAULT_BATCH_SIZE, MAX_BATCH_SIZE, predict_length_batched
from .policy import INFERENCE_PACKAGES, MODEL_ID, MODEL_REVISION
from .questions import validate_question

MODEL_DIR = f"/models/{MODEL_REVISION}"

app = modal.App("distill-jev-decisions")
optimization_app = modal.App("distill-jev-decisions-optimization")
volume = modal.Volume.from_name("distill-jev-decision-models", create_if_missing=True)
download_image = modal.Image.debian_slim(python_version="3.13").pip_install(
    "huggingface-hub==1.31.0"
)
# Inference dependencies from this checkpoint's requirements.txt. Training and
# HTTP-server dependencies are unnecessary for native Modal method calls.
inference_image = download_image.pip_install(*INFERENCE_PACKAGES).env(
    {"HF_HUB_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false"}
)

# No upstream wheel matches Torch 2.14 / CPython 3.13. Build the official
# release against our exact Torch ABI with a matching CUDA development toolkit.
optimization_image = (
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


def validate_rows(rows: list[dict[str, Any]], batch_size: int, *,
                  max_batch_size: int = MAX_BATCH_SIZE) -> None:
    """Validate small RPC inputs before allocating remote resources."""
    if type(batch_size) is not int or not 1 <= batch_size <= max_batch_size:
        raise ValueError(f"batch_size must be an integer between 1 and {max_batch_size}")
    if not isinstance(rows, list) or not 1 <= len(rows) <= 128:
        raise ValueError("Provide between 1 and 128 decisions per call")
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or "state" not in row:
            raise ValueError(f"Row {index} requires a state")
        question = row.get("question")
        if not isinstance(question, dict):
            raise ValueError(f"Row {index} requires a question object")
        validate_question(question)
        images = row.get("images", [])
        if not isinstance(images, list) or any(
            not isinstance(image, str) or not image.startswith("data:image/")
            for image in images
        ):
            raise ValueError("RPC images must be data:image/...;base64 URLs")
    # Reject non-JSON inputs and NaN before serialization to Modal.
    json.dumps(rows, allow_nan=False)


def _ensure_checkpoint() -> str:
    """Use a validated local completion manifest before consulting the Hub."""
    volume.reload()
    checkpoint = Path(MODEL_DIR)
    marker = checkpoint / ".download-complete.json"
    if marker.exists():
        saved = json.loads(marker.read_text())
        if saved.get("revision") == MODEL_REVISION and saved.get("files") and all(
            (checkpoint / name).is_file() and (checkpoint / name).stat().st_size == size
            for name, size in saved["files"].items()
        ):
            log("Checkpoint cache verified; skipping Hugging Face download and Volume commit")
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
    # The marker is only published after the downloaded files have been committed.
    with stage("Commit model Volume"):
        volume.commit()
    marker.write_text(json.dumps({"revision": MODEL_REVISION, "files": sizes}))
    volume.commit()
    return MODEL_DIR


@app.function(image=download_image, volumes={"/models": volume}, timeout=3600)
def download_model() -> str:
    """Cache the public checkpoint and its exact inference source on CPU."""
    return _ensure_checkpoint()


@optimization_app.function(image=download_image, volumes={"/models": volume}, timeout=3600)
def download_experiment_model() -> str:
    return _ensure_checkpoint()


def load_native_model() -> tuple[Any, Any, float]:
    started = perf_counter()
    log("GPU container ready; initializing model")
    with stage("Reload model Volume"):
        volume.reload()
    checkpoint = Path(MODEL_DIR)
    config = json.loads((checkpoint / "decision_config.json").read_text())
    if config.get("attention_mode") != "noncausal_full_attention":
        raise ValueError("Unexpected checkpoint attention mode")
    if config.get("pooling") != "last":
        raise ValueError("Unexpected checkpoint pooling")
    sys.path.insert(0, str(checkpoint / "source" / "src"))
    with stage("Import native model and inference dependencies"):
        native = import_module("autojev.model")
    with stage("Load BF16 checkpoint and transfer to CUDA"):
        model = native.DecisionModel(
            checkpoint, device="cuda", max_image_pixels=512 * 512,
        )
    elapsed = perf_counter() - started
    log(f"Model ready; initialization={elapsed:.1f}s")
    return model, native.answer, elapsed


@app.cls(
    image=inference_image, gpu="H100!", cpu=8, memory=131072,
    volumes={"/models": volume}, max_containers=1,
    scaledown_window=60, timeout=1800,
)
class Decisions:
    """Separate baseline/legacy worker; public prediction uses OptimizationExperiment."""
    model: Any = None
    answer: Any = None
    load_seconds: float = 0.0

    @modal.enter()
    def load(self) -> None:
        self.model, self.answer, self.load_seconds = load_native_model()

    @modal.method()
    def benchmark(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        """Baseline: 20 warmup predictions, then 200 measured batch-one calls."""
        if len(rows) != 220:
            raise ValueError("Benchmark requires 20 warmup and 200 measured rows")
        for start in range(0, len(rows), 128):
            validate_rows(rows[start:start + 128], 1)
        torch = import_module("torch")
        torch.cuda.synchronize()
        warmup_started = perf_counter()
        with stage("Warmup (20 predictions, batch size 1)"):
            for index, row in enumerate(rows[:20]):
                self.model.predict([row], batch_size=1)
                torch.cuda.synchronize()
                log_count("Warmup", index + 1, 20, warmup_started)
        torch.cuda.synchronize()
        warmup_seconds = perf_counter() - warmup_started
        torch.cuda.reset_peak_memory_stats()
        results = []
        latencies = []
        log("Measurement starting: 200 predictions, batch size 1")
        started = perf_counter()
        last_report = started
        for index, row in enumerate(rows[20:]):
            before = perf_counter()
            scores = self.model.predict([row], batch_size=1)[0]
            torch.cuda.synchronize()
            latencies.append(perf_counter() - before)
            results.append({
                "state": row["state"], "probabilities": scores,
                "answer": self.answer(row["question"], scores),
            })
            if index == 0 or (index + 1) % 10 == 0 or perf_counter() - last_report >= 5:
                log_count("Measurement", index + 1, 200, started)
                last_report = perf_counter()
        torch.cuda.synchronize()
        elapsed = perf_counter() - started
        log(f"Measurement complete: {200 / elapsed:.2f} rows/s; "
            f"peak allocated GPU memory={torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")
        ordered = sorted(latencies)
        return {
            "model_id": MODEL_ID, "model_revision": MODEL_REVISION,
            "question": rows[0]["question"], "batch_size": 1,
            "gpu": torch.cuda.get_device_name(0),
            # TorchVersion is a str subclass whose pickle requires local torch.
            "torch_version": str(torch.__version__),
            "transformers_version": import_module("transformers").__version__,
            "load_seconds": self.load_seconds,
            "warmup_rows": 20, "warmup_seconds": warmup_seconds,
            "measured_rows": 200, "measured_seconds": elapsed,
            "rows_per_second": 200 / elapsed,
            "latency_p50_seconds": (ordered[99] + ordered[100]) / 2,
            "latency_p95_seconds": ordered[189],
            "peak_gpu_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
            "peak_gpu_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
            "latencies_seconds": latencies, "results": results,
        }

    @modal.method()
    def predict(
        self, rows: list[dict[str, Any]], batch_size: int = 8,
    ) -> list[dict[str, Any]]:
        """Return calibrated distributions and native answers in input order.

        Complete inputs exceeding 8,192 tokens are rejected by the native model.
        Batch 8 is the short-message default; use smaller batches for longer
        inputs or images, since safe capacity depends on sequence length.
        """
        validate_rows(rows, batch_size, max_batch_size=8)
        probabilities = self.model.predict(rows, batch_size=batch_size)
        return [
            {
                "probabilities": scores,
                "answer": self.answer(row["question"], scores),
                "model_id": MODEL_ID,
                "model_revision": MODEL_REVISION,
            }
            for row, scores in zip(rows, probabilities, strict=True)
        ]


def predict_decisions(
    rows: list[dict[str, Any]], *, batch_size: int = DEFAULT_BATCH_SIZE,
) -> list[dict[str, Any]]:
    """Start a paid Modal run, cache weights, then score a small batch.

    No deployment or standalone CLI is required. Call only after reviewing the
    GPU configuration and supplying the intended decision query.
    Uses the validated optimized image, unquantized BF16 and length-sorted batch
    32 by default. A 16,384 padded-token budget and CUDA OOM splitting may reduce
    actual batch sizes. Complete inputs are never truncated; output order and
    native calibration are preserved. Explicitly lower batch_size if desired.
    """
    validate_rows(rows, batch_size)
    with modal.enable_output(), optimization_app.run():
        download_experiment_model.remote()
        return OptimizationExperiment().predict.remote(rows, batch_size=batch_size)


@optimization_app.cls(
    image=optimization_image, gpu="H100!", cpu=8, memory=131072,
    volumes={"/models": volume}, max_containers=1,
    scaledown_window=60, timeout=1800,
)
class OptimizationExperiment:
    """Optimized BF16 serving and opt-in benchmarks in the same single worker."""
    model: Any = None
    answer: Any = None
    load_seconds: float = 0.0
    kernel_validation: Any = None

    @modal.enter()
    def load(self) -> None:
        from .experiment import convolution_backends, validate_convolution_cuda

        # Execute CUDA first: wrapper inspection alone cannot establish dispatch.
        self.kernel_validation = validate_convolution_cuda((1, 2, 4, 8, 16, 32))
        convolution_backends(
            import_module("transformers.models.qwen3_5.modeling_qwen3_5"),
            import_module("causal_conv1d"),
        )
        self.model, self.answer, self.load_seconds = load_native_model()

    @modal.method()
    def predict(self, rows: list[dict[str, Any]],
                batch_size: int = DEFAULT_BATCH_SIZE) -> list[dict[str, Any]]:
        """Length-sort complete inputs, cap memory, restore order, retain native scores."""
        validate_rows(rows, batch_size)
        probabilities, stats = predict_length_batched(self.model, rows, batch_size)
        log(f"Prediction complete: {len(rows)} rows, batch ceiling={batch_size}, "
            f"runtime cap={stats['runtime_batch_cap']}, OOM splits={len(stats['oom_splits'])}")
        return [
            {
                "probabilities": scores,
                "answer": self.answer(row["question"], scores),
                "model_id": MODEL_ID,
                "model_revision": MODEL_REVISION,
            }
            for row, scores in zip(rows, probabilities, strict=True)
        ]

    @modal.method()
    def benchmark(self, rows: list[dict[str, Any]], settings: dict[str, Any] | None = None,
                  provenance: dict[str, Any] | None = None) -> dict[str, Any]:
        from .experiment import benchmark_batched_model, benchmark_loaded_model, validate_experiment_settings
        from uuid import uuid4

        for start in range(0, len(rows), 128):
            validate_rows(rows[start:start + 128], 1)
        if settings is not None:
            validate_experiment_settings(settings)
        remote_path = Path("/models/experiments") / f"causal-conv1d-bf16-{uuid4().hex}.json"
        remote_path.parent.mkdir(parents=True, exist_ok=True)
        log(f"Partial/final remote report: {remote_path}")

        def persist(report: dict[str, Any]) -> None:
            report["remote_report_path"] = str(remote_path)
            report["kernel_validation"] = self.kernel_validation
            payload = json.dumps(report, ensure_ascii=False, allow_nan=False)
            temporary = remote_path.with_suffix(".tmp")
            temporary.write_text(payload + "\n")
            temporary.replace(remote_path)
            with stage("Persist optimization progress to model Volume"):
                volume.commit()

        if settings is None:
            report = benchmark_loaded_model(
                self.model, self.answer, rows, self.load_seconds,
                model_id=MODEL_ID, model_revision=MODEL_REVISION,
            )
            persist(report)
        else:
            report = benchmark_batched_model(
                self.model, self.answer, rows, self.load_seconds,
                model_id=MODEL_ID, model_revision=MODEL_REVISION,
                settings=settings, persist=persist, provenance=provenance,
            )
        # Normalize package-specific subclasses before RPC (e.g. TorchVersion).
        return json.loads(json.dumps(report, ensure_ascii=False, allow_nan=False))


@optimization_app.function(image=optimization_image, gpu="H100!", cpu=2,
                           memory=8192, timeout=300, max_containers=1)
def validate_optimization_kernel() -> dict[str, Any]:
    """Paid tiny CUDA preflight, with no checkpoint Volume or model startup."""
    from .experiment import validate_convolution_cuda

    return validate_convolution_cuda()


def run_kernel_validation() -> dict[str, Any]:
    """Start only the paid kernel preflight; never load the checkpoint."""
    with modal.enable_output(), optimization_app.run():
        return validate_optimization_kernel.remote()
