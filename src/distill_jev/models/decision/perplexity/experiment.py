"""Single-worker convolution experiment; heavy imports stay remote."""

import inspect
import math
import statistics
from importlib import import_module, metadata
from time import perf_counter
from typing import Any
from unittest.mock import patch

from .progress import log, log_count, stage
from .batching import (
    plan_batches, predict_indexed_batch, rendered_input_lengths,
)


def compare_predictions(reference: list[dict[str, Any]],
                        candidate: list[dict[str, Any]]) -> dict[str, Any]:
    """Compare calibrated scores and native answers, not ground-truth accuracy."""
    if not reference:
        raise ValueError("Prediction comparison requires nonempty inputs")
    differences: list[float] = []
    changed: list[int] = []
    exact_answers = 0
    invalid_pairs: list[int] = []
    for index, (left, right) in enumerate(zip(reference, candidate, strict=True)):
        if left.get("error") or right.get("error"):
            invalid_pairs.append(index)
            continue
        differences.extend(abs(a - b) for a, b in zip(
            left["probabilities"], right["probabilities"], strict=True,
        ))
        left_answer, right_answer = left["answer"], right["answer"]
        exact_answers += left_answer == right_answer
        # Native choice answers include probabilities and confidence: comparing
        # the whole dict would mistake a numerical drift for a label flip.
        if isinstance(left_answer, dict) and left_answer.get("type") == "choice":
            same_decision = left_answer["choice"] == right_answer["choice"]
        else:
            same_decision = left_answer == right_answer
        if not same_decision:
            changed.append(index)
    compared = len(reference) - len(invalid_pairs)
    return {
        "compared_valid_rows": compared,
        "invalid_pair_indices": invalid_pairs,
        "answer_agreement_rows": compared - len(changed),
        "answer_agreement_fraction": (compared - len(changed)) / compared if compared else None,
        "changed_answer_indices": changed,
        "exact_native_answer_agreement_rows": exact_answers,
        "max_absolute_probability_difference": max(differences) if differences else None,
        "mean_absolute_probability_difference": sum(differences) / len(differences) if differences else None,
    }


def convolution_backends(qwen: Any, causal: Any) -> tuple[Any, Any]:
    """Resolve the reference only; wrapper identity is not dispatch evidence."""
    selected = qwen.causal_conv1d_fn
    # kernels 0.16 turns a decorated function into an nn.Module. Its forward
    # closure retains the Transformers package-fallback wrapper as `func`.
    if hasattr(selected, "forward"):
        selected = inspect.getclosurevars(selected.forward).nonlocals.get("func")
        if selected is None:
            raise RuntimeError("Unrecognized kernels convolution forward wrapper")
    while hasattr(selected, "__wrapped__"):
        selected = selected.__wrapped__
    reference_function = inspect.unwrap(selected)
    if reference_function.__module__ != qwen.__name__:
        raise RuntimeError("Could not identify upstream PyTorch reference convolution")
    return reference_function, causal.causal_conv1d_fn


def validate_convolution_cuda(batch_sizes: tuple[int, ...] = (1, 2, 4, 8)) -> dict[str, Any]:
    """Execute tiny BF16 tensors, without downloading or loading model weights."""
    torch = import_module("torch")
    causal = import_module("causal_conv1d")
    interface = import_module("causal_conv1d.causal_conv1d_interface")
    qwen = import_module("transformers.models.qwen3_5.modeling_qwen3_5")
    if not torch.cuda.is_available():
        raise RuntimeError("Convolution validation requires CUDA")
    cases = []
    generator = torch.Generator(device="cuda").manual_seed(42)
    with torch.inference_mode(), stage("Tiny BF16 CUDA convolution validation (no weights)"):
        for batch_size in batch_sizes:
            # Match the model's transposed QKV layout, no bias, width four, SiLU.
            x = torch.randn(batch_size, 32, 64, device="cuda", dtype=torch.bfloat16,
                            generator=generator).transpose(1, 2)
            weight = torch.randn(64, 4, device="cuda", dtype=torch.bfloat16,
                                 generator=generator) * 0.1
            expected = torch.nn.functional.silu(torch.nn.functional.conv1d(
                x, weight.unsqueeze(1), padding=3, groups=64,
            )[..., :32])
            with patch.object(interface, "causal_conv1d_fwd_function",
                              wraps=interface.causal_conv1d_fwd_function) as forward:
                direct = causal.causal_conv1d_fn(x, weight, activation="silu")
                torch.cuda.synchronize()
                direct_calls = forward.call_count
                forward.reset_mock()
                native = qwen.causal_conv1d_fn(x, weight, activation="silu")
                torch.cuda.synchronize()
                native_calls = forward.call_count
            if not direct_calls:
                raise RuntimeError("Direct package call did not execute the CUDA forward")
            if not native_calls:
                raise RuntimeError("Package CUDA test passed, but native Transformers dispatch "
                                   "did not execute causal-conv1d CUDA forward")
            for output in (direct, native):
                if output.dtype != x.dtype or output.device != x.device:
                    raise RuntimeError("Convolution changed output dtype/device")
                torch.testing.assert_close(output, expected, rtol=0.02, atol=0.002)
            cases.append({"batch_size": batch_size, "shape": list(x.shape),
                          "stride": list(x.stride()), "direct_cuda_forward_calls": direct_calls,
                          "native_cuda_forward_calls": native_calls,
                          "max_absolute_difference": float((native.float() - expected.float()).abs().max())})
    report = {"status": "passed", "gpu": str(torch.cuda.get_device_name(0)),
              "dtype": "bfloat16", "rtol": 0.02, "atol": 0.002,
              "torch_version": str(torch.__version__), "torch_cuda_version": str(torch.version.cuda),
              "cases": cases, "model_weights_loaded": False}
    log(f"Tiny CUDA validation passed: {len(cases)} shapes; native CUDA dispatch observed")
    return report


def benchmark_loaded_model(model: Any, answer: Any, rows: list[dict[str, Any]],
                           load_seconds: float, *, model_id: str,
                           model_revision: str) -> dict[str, Any]:
    if len(rows) != 220:
        raise ValueError("Experiment requires exactly 20 warmup and 200 measured rows")
    torch = import_module("torch")
    qwen = import_module("transformers.models.qwen3_5.modeling_qwen3_5")
    causal = import_module("causal_conv1d")
    # Dispatch was tested with CUDA tensors before loading this model.
    reference_function, _ = convolution_backends(qwen, causal)
    if next(model.backbone.parameters()).dtype != torch.bfloat16:
        raise RuntimeError("Expected unquantized BF16 model")

    def measure(backend: str, batch_size: int) -> dict[str, Any]:
        label = f"{backend}, batch={batch_size}"
        torch.cuda.synchronize()
        warmup_started = perf_counter()
        # Exactly the same 20 warmup rows each time: ceil(20 / batch_size)
        # calls; batch=8 ends in a four-row call, also warming that shape.
        with stage(f"Warmup ({label}, 20 rows)"):
            for start in range(0, 20, batch_size):
                model.predict(rows[start:min(start + batch_size, 20)], batch_size=batch_size)
                torch.cuda.synchronize()
                log_count(f"Warmup {label}", min(start + batch_size, 20), 20, warmup_started)
        warmup_seconds = perf_counter() - warmup_started
        # Keep the warmed allocator: empty_cache here would measure fresh
        # allocation overhead rather than warm steady-state inference.
        torch.cuda.reset_peak_memory_stats()
        results, latencies = [], []
        started = last_report = perf_counter()
        for start in range(0, 200, batch_size):
            batch = rows[20 + start:20 + min(start + batch_size, 200)]
            before = perf_counter()
            scores = model.predict(batch, batch_size=batch_size)
            torch.cuda.synchronize()
            latencies.append(perf_counter() - before)
            for row, probabilities in zip(batch, scores, strict=True):
                item = {"state": row["state"], "probabilities": probabilities}
                try:
                    item["answer"] = answer(row["question"], probabilities)
                except ValueError as exc:
                    # Preserve failures, never repair/renormalize upstream scores.
                    # JSON has no NaN/Inf, so retain them as explicit strings.
                    item["probabilities"] = [value if math.isfinite(value) else str(value)
                                             for value in probabilities]
                    item["answer"] = None
                    item["error"] = str(exc)
                    log(f"Invalid native prediction {label}, measured row {len(results)}: {exc}")
                results.append(item)
            completed = start + len(batch)
            if start == 0 or completed % 10 == 0 or perf_counter() - last_report >= 5:
                log_count(f"Measurement {label}", completed, 200, started)
                last_report = perf_counter()
        torch.cuda.synchronize()
        elapsed = perf_counter() - started
        allocated = torch.cuda.max_memory_allocated() / 2**30
        log(f"Complete {label}: {200 / elapsed:.2f} rows/s; peak allocated={allocated:.2f} GiB")
        return {"backend": backend, "batch_size": batch_size, "warmup_rows": 20,
                "warmup_calls": (20 + batch_size - 1) // batch_size,
                "warmup_seconds": warmup_seconds, "measured_rows": 200,
                "measured_seconds": elapsed, "rows_per_second": 200 / elapsed,
                "peak_gpu_allocated_gib": allocated,
                "peak_gpu_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
                "invalid_prediction_indices": [i for i, item in enumerate(results) if item.get("error")],
                "batch_latencies_seconds": latencies, "results": results}

    # This worker has no concurrent methods. Patch only this model module's
    # convolution entry point, restore in finally via patch.object, and leave
    # FLA, SDPA, readout, pooling and calibration untouched. No cache is used,
    # so causal_conv1d_update is never called by these full-sequence predictions.
    with patch.object(qwen, "causal_conv1d_fn", reference_function):
        reference = measure("upstream_pytorch_reference", 1)
    optimized: list[dict[str, Any]] = []
    for batch_size in (1, 2, 4, 8):
        try:
            result = measure("causal_conv1d", batch_size)
        except torch.cuda.OutOfMemoryError as exc:
            torch.cuda.empty_cache()
            optimized.append({"batch_size": batch_size, "status": "out_of_memory",
                              "error": str(exc)})
            continue
        result["reference_parity"] = compare_predictions(reference["results"], result["results"])
        if optimized and "results" in optimized[0]:
            result["optimized_batch_one_parity"] = compare_predictions(
                optimized[0]["results"], result["results"],
            )
        result["speedup_vs_same_run_reference"] = result["rows_per_second"] / reference["rows_per_second"]
        optimized.append(result)
    return {
        "model_id": model_id, "model_revision": model_revision,
        "question": rows[0]["question"], "dtype": "bfloat16", "quantization": None,
        "gpu": str(torch.cuda.get_device_name(0)),
        "gpu_total_memory_gib": torch.cuda.get_device_properties(0).total_memory / 2**30,
        "torch_cuda_version": str(torch.version.cuda), "load_seconds": load_seconds,
        "package_versions": {name: str(metadata.version(name)) for name in (
            "torch", "torchvision", "transformers", "causal-conv1d",
            "flash-linear-attention", "kernels", "huggingface-hub",
        )},
        "reference_selection": "Scoped upstream unwrapped convolution from kernels forward closure; restored before optimized runs",
        "warmup_policy": "Same first 20 reservoir rows per configuration, contiguous batches; "
        "20/10/5/3 calls for batch 1/2/4/8 (final batch 8 warmup has 4 rows).",
        "memory_policy": "Keep warmed allocator; reset peak counters before each measured run. Reserved memory can carry over between configurations.",
        "quality_scope": "Prediction parity on finite, native-answer-valid pairs only; "
        "invalid pairs explicitly excluded and counted. No ground-truth labels or accuracy assessment.",
        "reference": reference, "optimized": optimized,
    }


def validate_experiment_settings(settings: dict[str, Any]) -> None:
    """Bound opt-in experimental settings before starting remote resources."""
    expected = {"batch_sizes", "trials", "min_trial_seconds", "max_trial_passes", "max_padded_tokens"}
    if set(settings) != expected:
        raise ValueError(f"Experiment settings must contain exactly {sorted(expected)}")
    sizes = settings["batch_sizes"]
    if not isinstance(sizes, (list, tuple)) or not sizes:
        raise ValueError("batch_sizes must be a nonempty sequence of distinct sizes")
    for size in sizes:
        plan_batches([], size, max_padded_tokens=settings["max_padded_tokens"])
    if len(set(sizes)) != len(sizes):
        raise ValueError("batch_sizes must be distinct")
    for key, maximum in (("trials", 5), ("max_trial_passes", 100)):
        if type(settings[key]) is not int or not 1 <= settings[key] <= maximum:
            raise ValueError(f"{key} must be an integer between 1 and {maximum}")
    seconds = settings["min_trial_seconds"]
    if type(seconds) not in (int, float) or not math.isfinite(seconds) or not 0 <= seconds <= 60:
        raise ValueError("min_trial_seconds must be finite and between 0 and 60")


def _parity_summary(checks: list[dict[str, Any]], row_count: int) -> dict[str, Any]:
    """Conservative per-row agreement across every repeated pass."""
    invalid = set().union(*(set(check["invalid_pair_indices"]) for check in checks))
    changed = set().union(*(set(check["changed_answer_indices"]) for check in checks)) - invalid
    valid_count = row_count - len(invalid)
    compared_pairs = sum(check["compared_valid_rows"] for check in checks)
    maxima = [check["max_absolute_probability_difference"] for check in checks
              if check["max_absolute_probability_difference"] is not None]
    return {
        "checked_passes": len(checks), "compared_valid_rows": valid_count,
        "invalid_pair_indices": sorted(invalid), "changed_answer_indices": sorted(changed),
        "answer_agreement_rows": valid_count - len(changed),
        "answer_agreement_fraction": (valid_count - len(changed)) / valid_count if valid_count else None,
        "max_absolute_probability_difference": max(maxima) if maxima else None,
        # All rows in this fixed-question experiment have the same option count.
        "mean_absolute_probability_difference": sum(
            (check["mean_absolute_probability_difference"] or 0) * check["compared_valid_rows"]
            for check in checks) / compared_pairs if compared_pairs else None,
    }


def benchmark_batched_model(model: Any, answer: Any, rows: list[dict[str, Any]],
                            load_seconds: float, *, model_id: str, model_revision: str,
                            settings: dict[str, Any], persist: Any = None,
                            provenance: dict[str, Any] | None = None) -> dict[str, Any]:
    """Repeated fixed-sample trials; partial reports survive later config failures."""
    validate_experiment_settings(settings)
    if len(rows) != 220 or any(
        row["question"] != rows[0]["question"] or
        (row["question"]["type"] == "choice" and
         list(row["question"]["criteria"]) != list(rows[0]["question"]["criteria"]))
        for row in rows
    ):
        raise ValueError("Experiment requires 220 rows with the same question and candidate order")
    torch = import_module("torch")
    qwen = import_module("transformers.models.qwen3_5.modeling_qwen3_5")
    reference_function, _ = convolution_backends(qwen, import_module("causal_conv1d"))
    if next(model.backbone.parameters()).dtype != torch.bfloat16:
        raise RuntimeError("Expected unquantized BF16 model")
    report: dict[str, Any] = {
        "model_id": model_id, "model_revision": model_revision, "question": rows[0]["question"],
        "dtype": "bfloat16", "quantization": None, "status": "running", "settings": settings,
        "gpu": str(torch.cuda.get_device_name(0)),
        "gpu_total_memory_gib": torch.cuda.get_device_properties(0).total_memory / 2**30,
        "torch_cuda_version": str(torch.version.cuda), "load_seconds": load_seconds,
        "package_versions": {name: str(metadata.version(name)) for name in (
            "torch", "torchvision", "transformers", "causal-conv1d",
            "flash-linear-attention", "kernels", "huggingface-hub")},
        "warmup_policy": "First 20 rows plus one complete untimed pass over the same 200 measured rows, per configuration; warms actual batch/length shapes including batch 32.",
        "memory_policy": "Stable length sort, padded-token activation proxy (not guaranteed VRAM); halve on CUDA OOM and retain reduced runtime cap. Empty allocator only after OOM.",
        "timing_scope": "Sum of complete 200-row pass intervals including native preprocessing, transfers, answer conversion, synchronization, OOM recovery and progress. Excludes length scan, planning, warmup, parity comparison, persistence, loading and RPC.",
        "quality_scope": "Every measured pass checked against first reference pass, original row order; no ground-truth accuracy assessment. First-pass predictions stored for each trial.",
        "reference": None, "optimized": [], "provenance": provenance or {},
    }

    def save() -> None:
        if persist is not None:
            persist(report)

    canonical = None
    measured_rows = rows[20:]

    def measure(config: dict[str, Any], lengths: list[int]) -> None:
        nonlocal canonical
        size, sorted_by_length = config["batch_size"], config["length_sorted"]
        label = f"{config['backend']}, batch={size}, length_sorted={sorted_by_length}"
        plan = plan_batches(lengths[20:], size, length_sorted=sorted_by_length,
                            max_padded_tokens=settings["max_padded_tokens"])
        padded = sum(len(indices) * max(lengths[20 + i] for i in indices) for indices in plan)
        config.update({"planned_batch_sizes": [len(indices) for indices in plan],
                       "planned_padded_tokens_per_pass": padded,
                       "padding_fraction": 1 - sum(lengths[20:]) / padded,
                       "trials": [], "runtime_batch_cap": size})

        def run_pass(inputs: list[dict[str, Any]], batches: list[list[int]],
                     stats: dict[str, Any]) -> list[dict[str, Any]]:
            results: list[Any] = [None] * len(inputs)
            completed = 0
            for indices in batches:
                start = 0
                while start < len(indices):
                    cap = config["runtime_batch_cap"]
                    split_before = len(stats["oom_splits"])
                    indexed = predict_indexed_batch(model, inputs, indices[start:start + cap], torch, stats)
                    for event in stats["oom_splits"][split_before:]:
                        config["runtime_batch_cap"] = min(config["runtime_batch_cap"], event["batch_size"] // 2)
                    for index, scores in indexed:
                        item = {"state": inputs[index]["state"], "probabilities": scores}
                        try:
                            item["answer"] = answer(inputs[index]["question"], scores)
                        except ValueError as exc:
                            item.update({"probabilities": [value if math.isfinite(value) else str(value)
                                                            for value in scores],
                                         "answer": None, "error": str(exc)})
                        results[index] = item
                    completed += len(indexed)
                    start += len(indexed)
                # Sparse progress logging is part of measured end-to-end timing.
                if completed == len(inputs):
                    log(f"Pass complete ({label}): {completed} rows")
            if any(item is None for item in results):
                raise RuntimeError("Batch scheduler did not return every original row")
            return results

        def new_stats() -> dict[str, Any]:
            return {"successful_batch_sizes": [], "batch_latencies_seconds": [], "oom_splits": []}

        torch.cuda.synchronize()
        before = perf_counter()
        warmup_stats = new_stats()
        with stage(f"Warmup ({label}, 20 rows + 200 shape-warmup rows)"):
            warmup_plan = plan_batches(lengths[:20], size, length_sorted=sorted_by_length,
                                       max_padded_tokens=settings["max_padded_tokens"])
            run_pass(rows[:20], warmup_plan, warmup_stats)
            run_pass(measured_rows, plan, warmup_stats)
        config.update({"warmup_rows": 220, "warmup_seconds": perf_counter() - before,
                       "warmup_stats": warmup_stats, "status": "measuring"})
        save()
        checks = []
        for trial_index in range(settings["trials"]):
            torch.cuda.reset_peak_memory_stats()
            trial: dict[str, Any] = {"trial": trial_index + 1, "measured_seconds": 0.0, "measured_rows": 0,
                     "passes": [], **new_stats()}
            for pass_index in range(settings["max_trial_passes"]):
                torch.cuda.synchronize()
                before = perf_counter()
                results = run_pass(measured_rows, plan, trial)
                torch.cuda.synchronize()
                elapsed = perf_counter() - before
                trial["measured_seconds"] += elapsed
                trial["measured_rows"] += len(measured_rows)
                if canonical is None:
                    canonical = results
                parity = compare_predictions(canonical, results)
                checks.append(parity)
                trial["passes"].append({"pass": pass_index + 1, "measured_seconds": elapsed,
                                        "reference_parity": parity})
                if pass_index == 0:
                    trial["results"] = results
                if trial["measured_seconds"] >= settings["min_trial_seconds"]:
                    break
            trial.update({"rows_per_second": trial["measured_rows"] / trial["measured_seconds"],
                          "minimum_duration_reached": trial["measured_seconds"] >= settings["min_trial_seconds"],
                          "peak_gpu_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                          "peak_gpu_reserved_gib": torch.cuda.max_memory_reserved() / 2**30})
            config["trials"].append(trial)
            config["reference_parity"] = _parity_summary(checks, len(measured_rows))
            log(f"Trial {trial_index + 1} ({label}): {trial['rows_per_second']:.2f} rows/s, "
                f"{trial['measured_seconds']:.1f}s, peak={trial['peak_gpu_allocated_gib']:.2f} GiB")
            save()
        trials = config["trials"]
        config.update({"status": "completed", "unique_measured_rows": len(measured_rows),
                       "measured_rows": sum(trial["measured_rows"] for trial in trials),
                       "measured_seconds": sum(trial["measured_seconds"] for trial in trials),
                       "peak_gpu_allocated_gib": max(trial["peak_gpu_allocated_gib"] for trial in trials),
                       "peak_gpu_reserved_gib": max(trial["peak_gpu_reserved_gib"] for trial in trials),
                       "median_trial_rows_per_second": statistics.median(trial["rows_per_second"] for trial in trials),
                       "results": trials[0]["results"]})
        config["rows_per_second"] = config["measured_rows"] / config["measured_seconds"]
        if report["reference"] is not config:
            config["speedup_vs_same_run_reference"] = config["rows_per_second"] / report["reference"]["rows_per_second"]

    save()
    try:
        before = perf_counter()
        with stage("Scan full native rendered lengths (220 rows)"):
            lengths = rendered_input_lengths(model, rows)
        report.update({"length_scan_seconds": perf_counter() - before, "full_input_lengths": lengths,
                       "measured_input_tokens_per_pass": sum(lengths[20:])})
        reference = {"backend": "upstream_pytorch_reference", "batch_size": 1,
                     "length_sorted": False, "status": "warming"}
        report["reference"] = reference
        save()
        with patch.object(qwen, "causal_conv1d_fn", reference_function):
            measure(reference, lengths)
        save()
        configurations = [(1, False), (8, False)] + [(size, True) for size in settings["batch_sizes"]]
        for size, sorted_by_length in configurations:
            config = {"backend": "causal_conv1d", "batch_size": size,
                      "length_sorted": sorted_by_length, "status": "warming"}
            report["optimized"].append(config)
            save()
            try:
                measure(config, lengths)
            except torch.cuda.OutOfMemoryError as exc:
                config.update({"status": "out_of_memory", "error": str(exc)})
            # Outside except: release failed tensors before allocator cleanup.
            if config["status"] == "out_of_memory":
                torch.cuda.empty_cache()
            save()
        report["status"] = "completed"
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        for config in [report["reference"], *report["optimized"]]:
            if config is not None and config["status"] in ("warming", "measuring"):
                config.update({"status": "failed", "error": error})
        report.update({"status": "failed", "error": error})
        save()
        raise
    save()
    return report
