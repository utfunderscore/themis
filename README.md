# themis

Dataset preparation and moderation inference. See [DeBERTa training](docs/deberta-training.md) for CPU and CUDA training instructions.

The [Java inference project](java/README.md) provides a Java 21 Gradle scaffold for ONNX inference.

After category inference, the pipeline prepares
`cache/combined_chats_inference_sample_100k_review_inputs.jsonl` and runs a second
Perplexity pass, saving results to
`cache/combined_chats_inference_sample_100k_review_decisions.jsonl`. Both passes
reuse completed output or resume checkpointed shards after a failure.

To prepare review inputs directly from existing predictions without running inference:

```python
from themis.pipeline import generate_review_dataset

generate_review_dataset(
    source="cache/combined_chats_inference_sample_100k_decisions.jsonl",
    destination="cache/combined_chats_inference_sample_100k_review_inputs.jsonl",
)
```

Each row carries an `allow` / `block_and_review` question informed by its predicted
category, with the original prediction record preserved under `source`. The second
pass adds its result under `decision`, retaining the first-pass result in
`source.decision`. Calling `pipeline()` can start paid Modal GPU inference for
either pass if its completed output is missing.
