"""Matched CPU FP32 inference benchmark; call benchmark_onnx from Python."""

import json
import platform
from pathlib import Path
from time import perf_counter
from typing import cast

import numpy as np
import numpy.typing as npt
import onnxruntime as ort
import polars as pl
import torch

from .dataset import LABELS
from .inference import Moderation


def benchmark_onnx(
    model_directory: str | Path,
    onnx_directory: str | Path,
    dataset: str | Path,
    *,
    sample_size: int = 128,
    batch_sizes: tuple[int, ...] = (1, 8, 32),
    threads: int = 6,
    repeats: int = 3,
) -> dict[str, object]:
    """Time tokenization, forward pass and both softmaxes, excluding model load.

    Both backends use the original slow tokenizer, identical batches and CPU
    threads. Alternate backend order each repeat after three warmup batches.
    Latencies are per batch; throughput counts messages. Parity is measured on
    every timed batch. This measures Python bindings, not Java/JNI overhead.
    """
    if min(sample_size, threads, repeats) <= 0 or not batch_sizes or min(batch_sizes) <= 0:
        raise ValueError("Sample size, threads, repeats and batch sizes must be positive")
    directory = Path(onnx_directory)
    metadata = json.loads((directory / "inference_config.json").read_text())
    texts = pl.read_parquet(dataset).sample(n=sample_size, seed=42)["content"].to_list()
    old_threads = torch.get_num_threads()
    torch.set_num_threads(threads)
    try:
        start = perf_counter()
        original = Moderation(model_directory, device="cpu", max_length=metadata["max_length"])
        original.model.float()
        torch_load = perf_counter() - start
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        start = perf_counter()
        session = ort.InferenceSession(str(directory / "model.onnx"), options,
                                       providers=["CPUExecutionProvider"])
        onnx_load = perf_counter() - start

        def predict(batch, backend):
            inputs = original.tokenizer(
                batch, padding=True, truncation=True, max_length=original.max_length,
                return_tensors="np" if backend == "onnx" else "pt",
                return_token_type_ids=False,
            )
            if backend == "onnx":
                logits = cast(npt.NDArray[np.float32], session.run(["logits"], dict(inputs))[0])
            else:
                with torch.inference_mode():
                    logits = original.model(**inputs).logits.numpy()
            probabilities = []
            for group in (logits[:, :len(LABELS)], logits[:, len(LABELS):]):
                scores = np.exp(group - group.max(axis=-1, keepdims=True))
                probabilities.append(scores / scores.sum(axis=-1, keepdims=True))
            return logits, probabilities

        results = []
        for batch_size in batch_sizes:
            batches = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]
            for backend in ("pytorch", "onnx"):
                for i in range(3):
                    predict(batches[i % len(batches)], backend)
            timings = {name: [] for name in ("pytorch", "onnx")}
            max_logit_error = max_probability_error = 0.0
            agreement = np.zeros(2, dtype=np.int64)
            for repeat in range(repeats):
                for batch in batches:
                    outputs = {}
                    order = ("pytorch", "onnx") if repeat % 2 == 0 else ("onnx", "pytorch")
                    for backend in order:
                        start = perf_counter()
                        outputs[backend] = predict(batch, backend)
                        timings[backend].append(perf_counter() - start)
                    left, right = outputs["pytorch"], outputs["onnx"]
                    if not all(np.isfinite(output[0]).all() for output in outputs.values()):
                        raise ValueError("Non-finite benchmark logits")
                    max_logit_error = max(max_logit_error, float(np.abs(left[0] - right[0]).max()))
                    for i, (a, b) in enumerate(zip(left[1], right[1], strict=True)):
                        max_probability_error = max(max_probability_error, float(np.abs(a - b).max()))
                        agreement[i] += int((a.argmax(-1) == b.argmax(-1)).sum())
            stats = {}
            for backend, times in timings.items():
                stats[backend] = {
                    "messages_per_second": len(texts) * repeats / sum(times),
                    "batch_p50_ms": float(np.percentile(times, 50) * 1000),
                    "batch_p95_ms": float(np.percentile(times, 95) * 1000),
                    "timed_batches": len(times),
                }
            results.append({
                "batch_size": batch_size, **stats,
                "onnx_speedup": sum(timings["pytorch"]) / sum(timings["onnx"]),
                "max_absolute_logit_error": max_logit_error,
                "max_absolute_probability_error": max_probability_error,
                "category_agreement": float(agreement[0] / (len(texts) * repeats)),
                "action_agreement": float(agreement[1] / (len(texts) * repeats)),
            })
        lengths = [len(original.tokenizer(text, truncation=True, max_length=original.max_length)["input_ids"])
                   for text in texts]
        return {
            "platform": platform.platform(), "torch": torch.__version__, "onnxruntime": ort.__version__,
            "model_directory": str(model_directory), "onnx_directory": str(directory),
            "dataset": str(dataset), "sample_size": len(texts), "seed": 42,
            "threads": threads, "repeats": repeats, "dtype": "float32", "device": "cpu",
            "max_length": original.max_length,
            "token_length_p50_p95_max": np.percentile(lengths, [50, 95, 100]).tolist(),
            "load_seconds": {"pytorch": torch_load, "onnx": onnx_load},
            "results": results,
        }
    finally:
        torch.set_num_threads(old_threads)
