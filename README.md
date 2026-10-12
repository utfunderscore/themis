# distill-jev

## Perplexity decisions on Modal

`distill_jev.models.decision.perplexity.decisions` wraps the official `pplx-decider-v1.1-27b`
implementation at revision `6195aa55a72ae21e503c8a60198f7b390d42adb5`.
Importing it does not start remote work. **Calling `predict_decisions` starts
a paid Modal run**, builds the images, downloads approximately 49 GiB of weights
to a persistent Volume, and starts one H100 worker with 128 GiB CPU RAM.
Modal authentication and billing must already be configured.

```python
from distill_jev.models.decision.perplexity.decisions import predict_decisions

# Run only when ready to start Modal inference.
results = predict_decisions([{
    "state": "My integration keeps failing. Please help.",
    "question": {
        "type": "choice",
        "instructions": "Which team should handle this request?",
        "criteria": {
            "billing": "Charges and refunds",
            "support": "Technical integration errors",
            "sales": "Questions about buying a product",
        },
    },
}])
```

Results preserve input order and contain native answers, ordered candidate
probabilities, and model provenance. Supports `choice`, `score`, and `noul`
questions. Requests accept 1–128 decisions; public inference uses the validated
optimized image and defaults to **length-sorted batch 32**, using unquantized BF16.
The interface accepts batch ceilings 1–32. A **16,384 padded-token budget** limits
actual batches; CUDA OOMs are split and the reduced cap is retained for the rest
of the call. Singleton OOMs fail explicitly rather than dropping a row. Explicitly
lower `batch_size` if desired; short-message tests do not establish safe capacity
for all long inputs/images. Inputs exceeding 8,192 tokens including the complete
question/options/chat template/image tokens are rejected, never truncated.
Images must be base64 data URLs and use the evaluated 512×512 area cap.
The native BF16 implementation preserves noncausal full attention, last-token
pooling, and saved calibration. No KV caching or quantization is enabled.

This is the initial small-batch interface. Bulk dataset orchestration and
performance tuning should follow GPU correctness and memory measurements.
The promoted settings reuse the cached CUDA build and pinned checkpoint.

### Decision baseline benchmark

`distill_jev.models.decision.perplexity.benchmark.run_baseline(source, destination)`
starts a paid Modal run with the saved offensive-content category question. It
reservoir-samples 220 messages from a JSONL `content` field with seed 42, uses
20 for warmup, then measures 200 batch-size-one predictions on one H100.
The JSON report includes predictions, per-message latency, throughput, peak
GPU memory, model-loading time, and full remote-run wall time. Warm throughput
includes preprocessing and device transfers; it excludes loading and warmup.
Existing destination files are rejected to preserve previous measurements.
Timestamped logs show sampling progress, the Modal app ID, checkpoint download,
model initialization, each warmup prediction, and measured progress with rows/s
and ETA. Long setup stages emit a heartbeat every 15 seconds. Measured progress
prints every 10 rows or after 5 seconds at the next completed prediction.

### BF16 convolution optimization experiment

`distill_jev.models.decision.perplexity.benchmark.run_optimization(source, destination)`
starts a **paid** single-H100 experiment with the same seed-42 reservoir sample:
first 20 rows warmup, next 200 measured. One model load is shared by a batch-one
PyTorch convolution reference, optimized batch-one and contiguous batch-eight
controls, and length-sorted batches **8, 16, 32**. Public prediction now uses the
validated length-sorted batch-32 policy on this same optimized worker/image.
No quantization, attention replacement, prompting, pooling, readout or calibration
changes are made. The reference pass temporarily binds only Transformers'
upstream unwrapped convolution function in the single-request worker, restoring
the installed-package wrapper before optimized measurements. Before model loading,
tiny BF16 CUDA tensors test batch sizes 1/2/4/8/16/32 against a PyTorch reference and
observe calls to the installed package's CUDA forward through native Transformers
dispatch. Wrapper identity is not treated as proof of kernel availability.

To run only this **paid** preflight, without downloading or loading the checkpoint:

```python
from distill_jev.models.decision.perplexity.decisions import run_kernel_validation

report = run_kernel_validation()
```

The separate experiment image builds official `causal-conv1d==1.7.0` from source
against pinned Torch 2.14.0 / Python 3.13 with NVIDIA CUDA 13.0.2 development tools
(`--no-build-isolation`, `--no-deps`, `CAUSAL_CONV1D_FORCE_BUILD=TRUE`).
The installed GCC/G++ toolchain is selected explicitly (`CC=gcc`, `CXX=g++`)
because Torch's default compiler selection otherwise picked missing `clang++`.
No matching
Torch-2.14 release wheel was available when inspected. Official compatibility
sources: [release setup.py](https://github.com/Dao-AILab/causal-conv1d/blob/v1.7.0/setup.py)
(CUDA 13 and Torch ABI selection),
[README](https://github.com/Dao-AILab/causal-conv1d/blob/v1.7.0/README.md)
(BF16 support), and
[Transformers fallback selection](https://github.com/huggingface/transformers/blob/v5.17.0/src/transformers/integrations/hub_kernels.py).
The distinct baseline benchmark image is unchanged.

Native singleton `prepare()` calls count each complete rendered input, including
question, candidate descriptions, chat template and image tokens. Stable sorting
groups similar lengths; predictions are restored to original sample order before
comparison. A default **16,384 padded-token budget** limits batch activation size.
This is a conservative proxy, not a VRAM guarantee. CUDA OOMs are recorded and
split into smaller batches; the reduced runtime cap is retained. A singleton OOM
marks the configuration as failed rather than dropping its row.

Each configuration warms the same 20 messages **plus one untimed pass over all
200 measured rows**, warming the actual batch/length shapes, including batch 32.
It then runs **three trials**, each repeating the identical 200-row sample until
at least **15 measured seconds**, with a safety cap of 20 passes per trial.
Settings are configurable keyword arguments on `run_optimization`. The report
explicitly marks trials that hit the pass cap before reaching the target duration.
Repetition improves timing stability; it does **not** increase sample diversity.

Warm measurements retain the CUDA allocator cache and reset peak counters only;
cache clearing happens only after OOM. Timing includes native preprocessing,
transfers, native answer conversion, synchronization, OOM recovery and progress.
The one-time full-length scan is reported separately and excluded, as are planning,
warmup, parity comparisons, persistence, loading and RPC. Thus these are warm
inference rates, not whole-dataset wall-clock throughput. Batch latency is not
per-message latency.

Reports preserve first-pass predictions for each trial, package versions, repeated
pass timings, planned/effective batch sizes, padding, memory and parity metrics.
Every measured pass is checked against the same-run batch-one reference. Aggregate
agreement excludes rows invalid in any pass and counts a label change in any pass
as a disagreement. Probability drift includes every valid comparison, not just
the stored first-pass predictions. These are **numerical parity checks, not labeled
accuracy measurements**. BF16 batching and padding can change probabilities and
labels. Timestamped setup/warmup/progress logging remains enabled, and existing
local report files are never overwritten.
If the unchanged native answer converter rejects a score vector, that prediction
is saved with an error and no answer; nonfinite values are represented as strings
(`"nan"`, `"inf"`) for strict JSON. Invalid pairs are excluded from parity metrics
and their indices/counts are reported explicitly. Throughput alone is not a
correctness endorsement when any predictions are invalid.
The worker commits a uniquely named partial JSON report under `/models/experiments`
before warmup, after each trial/configuration and on failure. The same report is
atomically updated; completed configurations remain available if later work fails.
It includes source indices and run provenance before returning JSON-normalized
builtin values over RPC.

Preliminary 200-row throughput measurements on 2026-10-11, one H100 80GB,
pinned checkpoint, BF16 (not production-scale performance results):

| Configuration | Rows/s | 200-row time | Peak allocated GiB |
| --- | ---: | ---: | ---: |
| Same-run PyTorch reference, batch 1 | 10.48 | 19.09 s | 48.69 |
| Optimized convolution, batch 1 | 10.39 | 19.24 s | 48.69 |
| Optimized convolution, batch 2 | 19.93 | 10.04 s | 48.75 |
| Optimized convolution, batch 4 | 33.32 | 6.00 s | 48.88 |
| Optimized convolution, batch 8 | 36.32 | 5.51 s | 49.11 |

Report: `cache/perplexity_causal_conv1d_bf16_batches_200.json`. All 200 category
labels matched the reference in every optimized configuration, with no invalid
predictions or OOMs in the completed run. Maximum calibrated-probability drift
versus the reference was 0.02151 / 0.01579 / 0.01536 / 0.02266 for batches
1 / 2 / 4 / 8; this is not bitwise parity or a labeled accuracy evaluation.
The kernel alone did **not** improve measured batch-one throughput; batch eight
was 3.47× the same-run batch-one reference, reflecting batching, not an isolated
kernel speedup. The older 8.38 rows/s baseline used a different run and should
not be used to attribute a kernel-only improvement.

Native model loading in the successful run took 26.50 s; remote wall time was
167.99 s, excluding earlier attempts and the cached source-image build. Warmups
took 38.62 s for the reference and 1.98 / 1.78 / 1.37 / 9.51 s for optimized
batches 1 / 2 / 4 / 8. Shared compilation/cache state makes warmup durations
non-independent. One preceding attempt encountered a nonfinite reference score
vector; this did not recur in the successful run, and its cause remains
unexplained. The validity reporting exposes failures but does not fix them;
this small-sample result does not establish production numerical reliability.

### Repeated length-batching results

Report: `cache/perplexity_bf16_length_batched_repeated_200_2026-10-11.json`.
Same 200 unique measured messages/question/candidate order as the earlier reports;
three trials per configuration, each at least 15 measured seconds. The reference
probabilities were identical to the prior warm-cache report. Full rendered lengths
were 202–294 tokens across all 220 rows; the one-time length scan took 0.35 s.

| Configuration | Rows/s | Gain vs contiguous 8 | Peak allocated GiB | Labels agreeing | Max probability drift |
| --- | ---: | ---: | ---: | ---: | ---: |
| Same-run PyTorch reference, batch 1 | 7.12 | — | 48.69 | 200/200 | 0 |
| Optimized convolution, batch 1 | 7.11 | — | 48.69 | 200/200 | 0.02151 |
| Optimized, contiguous batch 8 | 35.13 | — | 49.11 | 200/200 | 0.02266 |
| Optimized, length-sorted batch 8 | 37.47 | +6.7% | 49.11 | 200/200 | 0.02151 |
| Optimized, length-sorted batch 16 | 39.98 | +13.8% | 49.38 | 199/200 | 0.01626 |
| Optimized, length-sorted batch 32 | 41.07 | +16.9% | 50.13 | 199/200 | 0.01918 |

Rates are total measured rows divided by total measured seconds across trials,
not median trial rates. Every measured pass was compared with the first reference
pass. All configurations completed without OOMs, cap reductions or invalid scores.
Sorted batch-eight padding fell from 7.29% to 1.20%. Batch-32 trial rates were
41.09 / 41.05 / 41.08 rows/s, over 19.5 measured seconds each. Its 12 measured
passes still contain only 200 unique messages; this is not expanded quality coverage.

Both larger-batch label changes occurred at exact ties in the reference distribution
and repeated in all trials: batch 16 changed measured row 99 from `insult` to `slur`,
and batch 32 changed measured row 30 from `none` to `insult`. There are no ground-truth
labels establishing which choice is correct. Mean absolute probability drift was
0.000513 / 0.000716 / 0.000648 / 0.000613 for contiguous 8 / sorted 8 / sorted 16 /
sorted 32; all-label agreement does not imply identical calibrated probabilities.

The cached image was reused and checkpoint Hub requests were skipped. Model loading
took 32.54 s; the entire remote job, including setup, warmups, persistence and app
shutdown, took 662.51 s (11.0 minutes). The app completed; no worker was retained.
The kernel alone again showed no batch-one improvement. With the user-approved
policy accepting a switch between exactly tied top reference labels, both batches
16 and 32 have **200/200 acceptable label agreement** on this sample. The table and
saved report above retain the original strict label-agreement numbers; no scores
or historical results were rewritten. Public inference now defaults to the fastest
tested policy: optimized BF16, length-sorted batch 32, 16,384 padded-token budget,
native calibration and original output order. Larger representative quality checks
and an explicit acceptable probability-drift policy are still needed, especially
before producing distillation targets.

### Next experiments before quantization

1. Build a resumable bulk pipeline that keeps one model loaded across chunks,
   preserves source IDs/order and calibrated probability vectors, and checkpoints
   progress without relabeling completed rows.
2. Validate the promoted path on a larger representative sample, including long
   inputs, unusual text and borderline categories. Report strict and exact-tie-aware
   label agreement separately, invalid scores, probability-drift distribution and
   labeled calibration/accuracy where ground truth is available.
3. Measure complete bulk wall time and cost per million messages, including the
   length scan, RPC, startup and checkpoint writes. Optimize startup/JIT reuse and
   compare GPU prices only after steady-state throughput is established.
4. Defer FP8/INT8 until bulk batching and probability-quality thresholds are settled.
   Confirm findings on more than 200 unique messages before treating them as
   production reliability evidence. Obtain approval before additional paid runs.

## Dataset preprocessing

### Modal decision startup notes

Baseline benchmarks and optimized serving/experiments use separate Modal apps.
Public `predict_decisions` uses the optimized app; `run_baseline` remains on the
baseline app/image. The compiled extension is an image layer and should be reused
when its build definition is unchanged. Checkpoint downloads publish a revision/file-size completion manifest;
subsequent runs validate it locally and skip Hub requests and Volume commits.
This checks completeness, not cryptographic integrity.

Warm benchmarks retain the CUDA allocator cache and reset peak counters only.
Reserved memory can carry over between configurations. Each `app.run()` is an
ephemeral job: keep related benchmarks in one run to amortize model initialization.
Increasing the idle timeout does not preserve a worker after its app exits.

The tiny CUDA preflight passed on 2026-10-11 on an H100 80GB with Torch
2.14.0+cu130: native Transformers dispatch executed the installed CUDA forward
for BF16 batches 1/2/4/8. Maximum absolute differences versus PyTorch were
0.00390625 / 0.001953125 / 0.0078125 / 0.00390625, within the elementwise
`rtol=0.02, atol=0.002` check. No checkpoint weights were loaded. This establishes
kernel availability and tiny-input dispatch/parity, not full-model correctness.
The earlier wrapper-inspection failure was not evidence of kernel unavailability.

### Common chat schema

Each dataset module provides `load(path)` and `download(output_path, force=False)`.
Both return a Polars `LazyFrame` with the same schema; `download` first caches the
raw dataset if needed:

| Field | Type | Minecraft Server Chat | 2b2t |
| --- | --- | --- | --- |
| `content` | string | `content` | `chat` |
| `username` | string or null | `username` | `player_name` |
| `date` | string or null | `date` | `time` |

Dates retain their original format: Minecraft has no explicit timezone; 2b2t
includes a UTC offset. They are not directly sortable across sources. Message
text is unchanged, and other source fields are omitted.

```python
from pathlib import Path
import polars as pl
from distill_jev.datasets.download_minecraft_server_chat import load as load_minecraft
from distill_jev.datasets.download_2b2t_chat import load as load_2b2t

chats = pl.concat([
    load_minecraft(Path("cache/minecraft_chats.jsonl")),
    load_2b2t(Path("2b2t-dataset/chats.csv")),
])
chats.unique(subset=["content"]).sink_ndjson("cache/combined_chats.jsonl")
```

Loading creates a lazy query, not a normalized file. Use `.collect()` to obtain a
DataFrame or `.sink_ndjson(path)` to write JSONL for moderation. Raw downloads are
not modified.

`distill_jev.datasets.deduplication.deduplicate_chats(source, destination)` uses Polars' streaming
engine to deduplicate JSONL by exact `content` text. It keeps the first record for
each message but does not preserve output order. Input rows are validated before
processing, and the destination is replaced only after successful completion.
Polars reserializes records into a shared schema; optional fields may become null.

Output order may differ between runs. To resume moderation checkpoints, reuse the
same deduplicated file with `moderate_dataset` rather than deduplicating again:
checkpoint identity includes the input file's bytes. Streaming deduplication still
requires state proportional to the number of distinct messages.

### Guided 100,000-chat inference sample

`distill_jev.pipeline.pipeline()` now adds a local sampling stage after completed
Koala moderation and writes `cache/combined_chats_inference_sample_100k.jsonl`.
It returns nothing and prints `Pipeline complete.` after all stages succeed.
It reuses `cache/combined_chats_moderated.jsonl` without rerunning moderation when
that completed file exists. If moderation is missing, the earlier pipeline stages
can still start paid Koala inference. This sampling stage itself never starts Modal
or Perplexity inference.

Approved quotas use the **winning Koala label**, not Perplexity categories:

| Label | Selected rows |
| --- | ---: |
| `OK` | 49,911 |
| `H` | 20,000 |
| `V` | 10,000 |
| `S` | 8,000 |
| `SH` | 7,000 |
| `HR` | 5,088 (all available) |
| `H2` | 1 (all available) |
| **Total** | **100,000** |

Within each category, select 20% with the smallest top-two probability gap first,
then the longest 10% of remaining chats by Unicode character count, then a uniform
seed-42 reservoir sample for the remaining 70%. The groups are disjoint and no
source row is selected twice. Fractional quotas are floored and the random group
receives the remainder: **19,999 ambiguous, 9,999 long, 70,002 random** overall.
Score-object insertion order breaks winning-label ties; earlier source row order
breaks ambiguity/length ranking ties. Each category has an independently seeded
random generator, so changing quota dictionary order does not change selection.
No `S3`/`V2` quota is assigned because neither label wins any existing predictions.

The external utility is reusable independently of the pipeline:

```python
from distill_jev.datasets.sampling import guided_sample_dataset

sample = guided_sample_dataset(
    "cache/combined_chats_moderated.jsonl",
    "cache/combined_chats_inference_sample_100k.jsonl",
    {"OK": 49_911, "H": 20_000, "V": 10_000, "S": 8_000,
     "SH": 7_000, "HR": 5_088, "H2": 1},
    seed=42,
    ambiguous_fraction=0.2,
    long_fraction=0.1,
)
```

Custom quotas and `text_field`, `score_field`, `metadata_field`, fractions and seed
are supported. Inputs must be scored JSONL records with finite probabilities.
All original fields and Koala probability values are preserved. The added
`sampling` object records a zero-based `source_row_index`, winning `category`,
selection `method`, `top_two_margin` and `content_characters`. Rows remain in source
order; no text is truncated, and character length is not a model token count.
Sampling is without replacement by source row; the input pipeline has already
deduplicated content. Koala scores are diversity signals, **not ground-truth labels**,
and this balanced/challenge-enriched sample does not estimate natural prevalence.

The `.jsonl.manifest.json` sidecar records the exact configuration, source/output
SHA-256 hashes, available counts and selected tier counts. Reuse verifies both
hashes and configuration; stale, mismatched or incomplete artifacts are rejected
rather than overwritten. Insufficient categories fail instead of silently moving
their quota. Choose a new destination when intentionally changing the sample.
Source JSONL is parsed once; a temporary 44-byte-per-eligible-row index supports
later selection passes. Memory scales with the requested sample, with temporary
index/output files on the destination filesystem cleaned up after completion.

## Dataset moderation on Modal

```sh
uv sync
uv run modal setup
```

Call one function from Python:

```python
from distill_jev.moderation import moderate_dataset

result = moderate_dataset(
    "cache/minecraft_chats.jsonl",
    "cache/minecraft_chats_moderated.jsonl",
    text_field="content",
    progress_interval=30,  # seconds between progress outputs
)
```

This automatically downloads the pinned `KoalaAI/Text-Moderation` weights to a
persistent Modal Volume, uploads the dataset as files, runs GPU inference, and
downloads the results to the returned local path. No deployment step is needed.

- Input: JSONL objects with a string `content` field (configurable with `text_field`).
- Output: original rows in original order, each with a `moderation` dictionary
  containing all nine label probabilities. An existing `moderation` field is rejected.
- Files are streamed and split into 10,000-row shards, suitable for millions of rows.
  Only file paths cross inference RPCs. Up to four T4 containers process shards;
  each loads FP16 weights once and processes 128 rows per GPU batch.
  Messages are tokenized once and bucketed by token length within each shard;
  FP32 softmax scores are restored to original row order before writing.
- `batch_size` accepts 1–128; `rows_per_shard` accepts 1–10,000. Each shard has a
  one-hour timeout. Texts are truncated to 512 tokens including special tokens.
- Allow local disk space for a temporary sharded input copy and the output.
  The destination is replaced only after the complete download succeeds.
- Job files use unique directories in `distill-jev-moderation-data` and are deleted
  after success. Failed jobs retain remote files for inspection; rerunning starts
  a new job. Keep the calling Python process running until completion.
- Weights remain cached in `distill-jev-moderation-models`. PyTorch and Transformers
  install only in the remote image.

Scores use softmax across `H`, `H2`, `HR`, `OK`, `S`, `S3`, `SH`, `V`, and `V2`
and sum to one per row. See the
[model card](https://huggingface.co/KoalaAI/Text-Moderation) for label definitions.

### Progress and sharding

During inference, progress prints every `progress_interval` seconds plus a final
summary: elapsed seconds, completed rows / total rows, and rows per second since
the previous output. Counts aggregate all workers, which publish updates after
scoring windows and at shard completion. Rates are therefore approximate.
Elapsed time is wall-clock time from inference submission, including GPU startup,
queueing and result-file commits; it excludes upload, model download and result
download. It is not the sum of GPU compute time across workers.

For 2,000,000 rows with the defaults:

1. The local JSONL is streamed into **200 numbered files of 10,000 rows** each.
   The last shard can be smaller for other dataset sizes.
2. Files are uploaded to a unique job directory in the data Volume.
3. Modal schedules one file per task across **up to four T4 workers**. Each worker
    processes its file in **128-row, length-bucketed FP16 GPU batches**, writes a scored file, then takes
   another shard. A shard is a scheduling unit; a batch is a GPU memory unit.
4. Progress includes partial shards even if an earlier shard is still running.
5. Once inference finishes, scored files are downloaded and concatenated in their
   original numbered order. Parallel completion order does not reorder rows.

Smaller shards give finer scheduling units; larger shards reduce per-file overhead.
`rows_per_shard` changes file/task size, while `batch_size` changes GPU batch size.
