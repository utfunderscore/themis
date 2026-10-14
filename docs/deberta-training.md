# DeBERTa-v3 fine-tuning

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

Run using the prepared splits in this repository (choose a new output directory):

```sh
uv run --no-sync python -c 'from distill_jev.models.deberta.training import train; train("cache/deberta/splits", "cache/deberta/run-001")'
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

The loss is soft-target categorical cross-entropy over `LABELS`, preserving the
teacher's full probability distribution. Dynamic padding reduces wasted compute.
The best epoch is selected using validation loss, then evaluated on the test split.
Agreement and per-class/macro F1 compare against teacher argmax labels; they are
not measurements against human annotations. KL measures distribution fidelity.

Outputs include `model/` (Hugging Face model and tokenizer), `checkpoints/`,
`run_config.json`, and `metrics.json`. For inference, load `model/` using
`AutoTokenizer` and `AutoModelForSequenceClassification`, use the same token limit,
and apply softmax to logits. Class order is recorded in the model configuration.
