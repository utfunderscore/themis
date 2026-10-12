"""Opt-in, paid baseline benchmark with reproducible local sampling."""

import json
import random
from pathlib import Path
from time import perf_counter
from typing import Any

import modal

from .decisions import (
    Decisions, OptimizationExperiment, app, optimization_app,
    download_model, download_experiment_model, validate_rows,
)
from .questions import CATEGORY_QUESTION
from .progress import log, stage
from .batching import DEFAULT_PADDED_TOKEN_BUDGET
from .experiment import validate_experiment_settings


def sample_benchmark_rows(source: str | Path) -> tuple[list[dict[str, Any]], list[int], int]:
    """Preserve the baseline reservoir algorithm and order exactly."""
    source = Path(source)
    rng = random.Random(42)
    sample = []
    total = 0
    log(f"Sampling {source}: seed=42, warmup=20, measured=200")
    sampling_started = perf_counter()
    last_report = sampling_started
    with source.open() as reader:
        for index, line in enumerate(reader):
            total += 1
            if perf_counter() - last_report >= 5:
                log(f"Sampling: scanned {total:,} rows; "
                    f"elapsed={perf_counter() - sampling_started:.1f}s")
                last_report = perf_counter()
            selected = index if index < 220 else rng.randrange(index + 1)
            if selected < 220:
                row = json.loads(line)
                if not isinstance(row.get("content"), str):
                    raise ValueError(f"Source row {index} requires string content")
                item = (index, row["content"])
                if index < 220:
                    sample.append(item)
                else:
                    sample[selected] = item
    if len(sample) != 220:
        raise ValueError("Source must have at least 220 messages")
    rows = [{"state": text, "question": CATEGORY_QUESTION} for _, text in sample]
    for start in range(0, len(rows), 128):
        validate_rows(rows[start:start + 128], 1)
    log(f"Sampling complete: scanned {total:,} rows; selected {len(sample)}")
    return rows, [index for index, _ in sample], total


def run_baseline(source: str | Path, destination: str | Path) -> dict[str, Any]:
    """Sample 220 messages: 20 warmup, 200 timed; save predictions and metrics."""
    source, destination = Path(source), Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    rows, indices, total = sample_benchmark_rows(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    log("Starting Modal: one H100, batch size 1, BF16")
    started = perf_counter()
    report = None
    with stage("Remote baseline (image build, download, startup, inference)"):
        with modal.enable_output(), app.run():
            log(f"Modal app ID: {app.app_id}")
            with stage("Ensure checkpoint is cached on Modal"):
                download_model.remote()
            log("Requesting GPU benchmark; waiting for scheduling/model initialization")
            report = Decisions().benchmark.remote(rows)
        if report is None:
            raise InterruptedError("Modal run ended before returning benchmark results")
    report.update({
        "source": str(source), "source_rows": total, "seed": 42,
        "sample_source_indices": indices,
        "remote_run_wall_seconds": perf_counter() - started,
        "timing_scope": "Warm batch-one prediction including preprocessing, transfers, "
        "native answer conversion and progress; excludes loading, warmup and RPC.",
    })
    with destination.open("x") as writer:
        json.dump(report, writer, indent=2, ensure_ascii=False, allow_nan=False)
        writer.write("\n")
    log(f"Saved report to {destination}; throughput={report['rows_per_second']:.2f} rows/s")
    return report


def run_optimization(source: str | Path, destination: str | Path, *,
                     batch_sizes: tuple[int, ...] = (8, 16, 32), trials: int = 3,
                     min_trial_seconds: float = 15.0, max_trial_passes: int = 20,
                     max_padded_tokens: int = DEFAULT_PADDED_TOKEN_BUDGET) -> dict[str, Any]:
    """Paid repeated fixed-sample experiment, one load; never overwrite reports.

    Public inference remains unchanged. Reference/optimized batch-one and an
    unsorted batch-eight control accompany the requested length-sorted sizes.
    """
    settings = {"batch_sizes": list(batch_sizes), "trials": trials,
                "min_trial_seconds": min_trial_seconds, "max_trial_passes": max_trial_passes,
                "max_padded_tokens": max_padded_tokens}
    validate_experiment_settings(settings)
    source, destination = Path(source), Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    rows, indices, total = sample_benchmark_rows(source)
    provenance = {"source": str(source), "source_rows": total, "seed": 42,
                  "sample_source_indices": indices}
    destination.parent.mkdir(parents=True, exist_ok=True)
    report = None
    started = perf_counter()
    with stage("Remote convolution optimization experiment"):
        with modal.enable_output(), optimization_app.run():
            app_id = str(optimization_app.app_id)
            log(f"Modal app ID: {app_id}")
            provenance["modal_app_id"] = app_id
            download_experiment_model.remote()
            report = OptimizationExperiment().benchmark.remote(rows, settings=settings, provenance=provenance)
        if report is None:
            raise InterruptedError("Modal run ended before returning optimization results")
    report.update({
        "source": str(source), "source_rows": total, "seed": 42,
        "sample_source_indices": indices, "modal_app_id": app_id,
        "remote_run_wall_seconds": perf_counter() - started,
    })
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x") as writer:
        json.dump(report, writer, indent=2, ensure_ascii=False, allow_nan=False)
        writer.write("\n")
    log(f"Saved optimization report to {destination}")
    return report
