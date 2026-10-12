"""Length-aware scheduling around the unchanged native DecisionModel predictor."""

from importlib import import_module
from time import perf_counter
from typing import Any

MAX_INPUT_TOKENS = 8192
MAX_BATCH_SIZE = 32
DEFAULT_BATCH_SIZE = 32
DEFAULT_PADDED_TOKEN_BUDGET = 16384


def rendered_input_lengths(model: Any, rows: list[dict[str, Any]]) -> list[int]:
    """Use native rendering/processing, including image tokens; never truncate.

    Singleton prepare calls briefly transfer inputs to CUDA, but do not run the
    backbone. This scan is setup work and is reported separately from throughput.
    """
    lengths = []
    for index, row in enumerate(rows):
        prepared = model.prepare([row], max_length=MAX_INPUT_TOKENS)
        length = int(prepared.input_tokens)
        del prepared
        if not 1 <= length <= MAX_INPUT_TOKENS:
            raise ValueError(f"Row {index} has invalid full input length {length}")
        lengths.append(length)
    return lengths


def plan_batches(lengths: list[int], batch_size: int, *, length_sorted: bool = True,
                 max_padded_tokens: int = DEFAULT_PADDED_TOKEN_BUDGET) -> list[list[int]]:
    """Stable length sorting with an activation-memory proxy, not a VRAM guarantee."""
    if type(batch_size) is not int or not 1 <= batch_size <= MAX_BATCH_SIZE:
        raise ValueError(f"batch_size must be an integer between 1 and {MAX_BATCH_SIZE}")
    if type(max_padded_tokens) is not int or max_padded_tokens < MAX_INPUT_TOKENS:
        raise ValueError("max_padded_tokens must allow one complete 8192-token input")
    if any(type(length) is not int or not 1 <= length <= MAX_INPUT_TOKENS for length in lengths):
        raise ValueError("Full input lengths must be integers between 1 and 8192")
    order = sorted(range(len(lengths)), key=lengths.__getitem__) if length_sorted else list(range(len(lengths)))
    batches, current = [], []
    longest = 0
    for index in order:
        next_longest = max(longest, lengths[index])
        if current and (len(current) == batch_size or
                        (len(current) + 1) * next_longest > max_padded_tokens):
            batches.append(current)
            current, longest = [], 0
        current.append(index)
        longest = max(longest, lengths[index])
    if current:
        batches.append(current)
    return batches


def predict_indexed_batch(model: Any, rows: list[dict[str, Any]], indices: list[int],
                          torch: Any, stats: dict[str, Any]) -> list[tuple[int, list[float]]]:
    """Split only CUDA OOMs; keep warmed allocator on every successful call.

    Leave the except scope before clearing the allocator/retrying so the failed
    prediction's traceback no longer retains its tensors. Singleton OOMs remain
    fatal to the configuration, rather than silently dropping an input.
    """
    started = perf_counter()
    try:
        scores = model.predict([rows[index] for index in indices], batch_size=len(indices))
        torch.cuda.synchronize()
    except torch.cuda.OutOfMemoryError as exc:
        if len(indices) == 1:
            raise
        failure = {"indices": list(indices), "batch_size": len(indices), "error": str(exc)}
    else:
        stats["successful_batch_sizes"].append(len(indices))
        stats["batch_latencies_seconds"].append(perf_counter() - started)
        return list(zip(indices, scores, strict=True))
    stats["oom_splits"].append(failure)
    torch.cuda.empty_cache()
    midpoint = len(indices) // 2
    return (predict_indexed_batch(model, rows, indices[:midpoint], torch, stats) +
            predict_indexed_batch(model, rows, indices[midpoint:], torch, stats))


def predict_length_batched(model: Any, rows: list[dict[str, Any]],
                           batch_size: int = DEFAULT_BATCH_SIZE) -> tuple[list[list[float]], dict[str, Any]]:
    """Serve the validated length-sorted BF16 policy without altering native scores.

    The requested size is a ceiling. Token-budget limits and CUDA OOM recovery
    may use smaller batches; a learned OOM cap lasts for the rest of this call.
    Scan every full input first, so overflow is rejected before any forward pass.
    """
    if not rows:
        raise ValueError("Provide at least one decision")
    # Validate the batch ceiling before touching the model or importing Torch.
    plan_batches([], batch_size)
    lengths = rendered_input_lengths(model, rows)
    batches = plan_batches(lengths, batch_size)
    torch = import_module("torch")
    stats: dict[str, Any] = {
        "full_input_lengths": lengths, "successful_batch_sizes": [],
        "batch_latencies_seconds": [], "oom_splits": [], "runtime_batch_cap": batch_size,
    }
    results: list[list[float] | None] = [None] * len(rows)
    for indices in batches:
        start = 0
        while start < len(indices):
            cap = stats["runtime_batch_cap"]
            splits_before = len(stats["oom_splits"])
            indexed = predict_indexed_batch(model, rows, indices[start:start + cap], torch, stats)
            for event in stats["oom_splits"][splits_before:]:
                stats["runtime_batch_cap"] = min(stats["runtime_batch_cap"], event["batch_size"] // 2)
            for index, scores in indexed:
                results[index] = scores
            start += len(indexed)
    if any(scores is None for scores in results):
        raise RuntimeError("Batch scheduler did not return every original row")
    return [scores for scores in results if scores is not None], stats
