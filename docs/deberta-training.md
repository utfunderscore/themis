# DeBERTa-v3 fine-tuning

## Training on Modal

The separate `distill_jev.models.deberta.modal_training` module runs the same
dual-target trainer on an H100 GPU (4 CPUs, 32 GiB RAM, 24-hour timeout).
Only the base project dependencies are needed locally. Authenticate once with
`uv run modal setup`, then call from Python with your prepared splits:

```python
from distill_jev.models.deberta.modal_training import train

result = train(
    "cache/deberta/review-splits",
    config={"bf16": True, "train_batch_size": 16},
)
print(result)
```

`config` accepts the fields of `TrainingConfig`; omitted fields use the local
trainer's defaults. The call uploads the three Parquet splits, waits for training,
and returns metrics and Volume-relative artifact paths. Each call creates a unique
run ID, printed before training starts. Keep the calling process running until it
finishes.

Models, checkpoints, and metrics persist in the `distill-jev-deberta-training`
Modal Volume under `runs/<run_id>/output`. Download a completed run with:

```sh
uv run modal volume get distill-jev-deberta-training runs/<run_id>/output cache/deberta/modal-run
```

Hugging Face downloads are cached in the same Volume. Failed runs preserve partial
artifacts where possible; another call starts a fresh run rather than resuming.
The automatic `pipeline()` training step uses Modal and records successful results
in `cache/deberta/review-modal-run.json`. Delete that receipt to start another run.
The local training call is retained as comments in `pipeline.py`.

## Local training

`pipeline()` automatically prepares dual-target splits in
`cache/deberta/review-splits` and trains on Modal after review scoring finishes.
Completed splits and the Modal run receipt are reused. Move the splits and receipt
aside to retrain when teacher results change. For manual local training, install
the training dependencies below.

Install the training dependencies with the PyTorch build for your machine (choose one):

```sh
uv sync --extra training --extra cpu   # CPU-only PyTorch
uv sync --extra training --extra cuda  # CUDA 12.8 PyTorch; requires an NVIDIA GPU and compatible driver
```

Verify the selected build with `uv run --no-sync python -c 'import torch; print(torch.__version__, torch.cuda.is_available())'`.
The CUDA command should print `True` for availability; a CPU build prints `False`.
Switching builds requires another `uv sync` with the desired extras. Do not enable
`cpu` and `cuda` together. Training selects CUDA automatically when available;
otherwise it runs on CPU. Set `use_cpu=True` to force CPU even with a CUDA build.

Regenerate splits from the pipeline's completed second-pass results (old category-only
splits are incompatible):

```python
from distill_jev.models.deberta import prepare_dataset

prepare_dataset(
    "cache/combined_chats_inference_sample_100k_review_decisions.jsonl",
    "cache/deberta/review-splits",
)
```

Run using those splits (choose a new output directory):

```sh
uv run --no-sync python -c 'from distill_jev.models.deberta.training import train; train("cache/deberta/review-splits", "cache/deberta/run-001")'
```

Or call from Python on your training machine:

```python
from distill_jev.models.deberta.training import TrainingConfig, train

result = train(
    "path/to/splits",  # train.parquet, validation.parquet, test.parquet
    "cache/deberta/run-001",  # new directory
    config=TrainingConfig(
        model_name="microsoft/deberta-v3-base",
        epochs=3,
        train_batch_size=16,
        gradient_accumulation_steps=2,
        bf16=False,  # set True only on hardware supporting BF16
    ),
)
```

Defaults use FP32, a 256-token limit, seed 42, AdamW, and a linear learning-rate
schedule with 10% warmup. Set `revision` to a Hub commit for reproducible weights.
Use `use_cpu=True` and small batches for CPU smoke runs. Each run must use a new
output directory; training will not overwrite an existing one.

The shared encoder predicts eight logits: six categories in `LABELS` order,
then `allow` and `block_and_review` in `ACTION_LABELS` order. The loss sums two
independent soft-target cross-entropies with equal weight, preserving both teacher
distributions. Category scores come from `source.decision`; action scores come
from the second-pass `decision`. Only the chat text is supplied to the student.
Dynamic padding reduces wasted compute.
The best epoch is selected using validation loss, then evaluated on the test split.
Agreement and per-class/macro F1 compare against teacher argmax labels; they are
not measurements against human annotations. KL measures distribution fidelity.
Metrics are reported separately with `category_` and `action_` prefixes.

Outputs include `model/` (Hugging Face model and tokenizer), `checkpoints/`,
`run_config.json`, and `metrics.json`. For inference, load `model/` using
`AutoTokenizer` (with `use_fast=False`) and `AutoModelForSequenceClassification`, use the same token limit,
and apply softmax **separately** to `logits[..., :6]` and `logits[..., 6:]`.
Never apply one softmax across all eight logits. Action argmax 1 means block for
human review; 0 means allow, independently of category. Class order and task
groups are recorded in the model configuration.
