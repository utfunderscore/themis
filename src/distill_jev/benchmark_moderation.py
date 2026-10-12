"""Importable, opt-in GPU experiments for the pinned moderation model."""

import json
from contextlib import nullcontext
from pathlib import Path
from time import perf_counter

import modal

from .moderation import MODEL_DIR, download_model, inference_image, volume

app = modal.App("distill-jev-moderation-benchmark")


@app.function(image=inference_image, gpu="T4", volumes={"/models": volume}, timeout=600)
def benchmark(texts: list[str]) -> list[dict]:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_DIR, local_files_only=True, use_safetensors=True
    ).cuda().eval()
    baseline = None
    reports = []
    configurations = [(False, "fp32", 16), (True, "fp32", 16)]
    configurations += [(True, "fp32", size) for size in (64, 256)]
    configurations += [(True, "fp16_weights", size) for size in (16, 32, 64, 128, 256)]
    for bucket, precision, size in configurations:
        if precision == "fp16_weights":
            model.half()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        # Warm kernels before timing; model startup is deliberately excluded.
        warm = tokenizer(texts[:size], padding=True, truncation=True,
                         max_length=512, return_tensors="pt").to("cuda")
        try:
            with torch.inference_mode():
                for _ in range(3):
                    model(**warm)
        except RuntimeError as exc:
            report = dict(bucket=bucket, precision=precision, batch_size=size, error=str(exc))
            reports.append(report)
            print(json.dumps(report), flush=True)
            del warm
            continue
        del warm
        torch.cuda.synchronize()
        started = perf_counter()
        output = torch.empty((len(texts), 9), dtype=torch.float32)
        padded_tokens = 0
        tokenize_seconds = 0.0
        try:
            with torch.inference_mode():
                for offset in range(0, len(texts), 10000):
                    window = texts[offset:offset + 10000]
                    before = perf_counter()
                    encoded = tokenizer(window, truncation=True, max_length=512) if bucket else None
                    indices = sorted(range(len(window)), key=lambda i: len(encoded["input_ids"][i])) if bucket else list(range(len(window)))
                    tokenize_seconds += perf_counter() - before
                    # Keep the autocast weight cache across all batches in a shard.
                    with torch.autocast("cuda") if precision == "fp16" else nullcontext():
                        for start in range(0, len(indices), size):
                            batch = indices[start:start + size]
                            before = perf_counter()
                            if bucket:
                                inputs = tokenizer.pad(
                                    [{key: value[i] for key, value in encoded.items()} for i in batch],
                                    padding=True, return_tensors="pt",
                                )
                            else:
                                inputs = tokenizer([window[i] for i in batch], padding=True,
                                                   truncation=True, max_length=512, return_tensors="pt")
                            tokenize_seconds += perf_counter() - before
                            padded_tokens += inputs["input_ids"].numel()
                            logits = model(**inputs.to("cuda")).logits
                            scores = logits.float().softmax(-1).cpu()
                            output[[offset + i for i in batch]] = scores
            torch.cuda.synchronize()
            elapsed = perf_counter() - started
            if not torch.isfinite(output).all():
                raise ValueError("Non-finite probabilities")
            if baseline is None:
                baseline = output.clone()
            difference = (output - baseline).abs()
            report = dict(bucket=bucket, precision=precision, batch_size=size,
                          rows=len(texts), seconds=elapsed, rows_per_second=len(texts)/elapsed,
                          tokenization_seconds=tokenize_seconds, padded_tokens=padded_tokens,
                          peak_gpu_mb=torch.cuda.max_memory_allocated()/2**20,
                          max_probability_difference=difference.max().item(),
                          mean_probability_difference=difference.mean().item(),
                          label_agreement=(output.argmax(-1) == baseline.argmax(-1)).float().mean().item())
        except (RuntimeError, ValueError) as exc:
            report = dict(bucket=bucket, precision=precision, batch_size=size, error=str(exc))
        reports.append(report)
        print(json.dumps(report), flush=True)
    return reports


def run_experiments(source: str | Path, destination: str | Path, sample_size: int = 1000) -> list[dict]:
    """Run paid T4 experiments using a reproducible uniform reservoir sample."""
    import random
    from .moderation import app as production_app

    if sample_size < 256:
        raise ValueError("sample_size must be at least 256")
    randomizer = random.Random(42)
    texts = []
    with Path(source).open() as reader:
        for index, line in enumerate(reader):
            selected = index if index < sample_size else randomizer.randrange(index + 1)
            if selected < sample_size:
                text = json.loads(line)["content"]
                if index < sample_size:
                    texts.append(text)
                else:
                    texts[selected] = text
    if not texts:
        raise ValueError("Input is empty")
    with modal.enable_output(), production_app.run():
        download_model.remote()
    with modal.enable_output(), app.run():
        reports = benchmark.remote(texts)
    Path(destination).write_text(json.dumps(reports, indent=2) + "\n")
    return reports
