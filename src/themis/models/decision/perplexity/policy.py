"""Pinned inference provenance, shared without importing Modal or Torch."""

MODEL_ID = "perplexity-ai/pplx-decider-v1.1-27b"
MODEL_REVISION = "6195aa55a72ae21e503c8a60198f7b390d42adb5"
PIPELINE_VERSION = 1
INFERENCE_PACKAGES = (
    "torch==2.14.0", "torchvision==0.29.0", "transformers==5.17.0",
    "accelerate==1.15.0", "pillow==12.3.0", "kernels==0.16.1",
    "flash-linear-attention==0.5.2", "safetensors==0.8.0", "numpy==2.5.3",
)
NUMERICAL_POLICY = {
    "weights": "unquantized-bfloat16",
    "predictor": "pinned-native-autojev",
    "probabilities": "native-calibrated-unmodified",
    "answer": "native-autojev.answer",
    "attention": "noncausal_full_attention",
    "pooling": "last",
    "truncation": "reject-overflow",
    "batching": "stable-length-sort-padded-token-budget-cuda-oom-split",
    "convolution": "causal-conv1d==1.7.0-native-cuda",
    "inference_packages": list(INFERENCE_PACKAGES),
}
