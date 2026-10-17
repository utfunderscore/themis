"""Local inference for the distilled student. Requires the ``training`` extra."""

import json
from pathlib import Path
from typing import TypedDict

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .dataset import ACTION_LABELS, LABELS


class Prediction(TypedDict):
    category: str
    action: str
    category_probabilities: dict[str, float]
    action_probabilities: dict[str, float]


class Moderation:
    """Load once, then predict batches on CUDA when available, otherwise CPU.

    ``model_directory`` accepts a training output directory or its ``model/``
    subdirectory (including downloaded Modal artifacts). The token limit comes
    from ``max_length``, else ``run_config.json``, else the saved tokenizer's
    ``model_max_length`` when it is within the model's position limit.
    """

    def __init__(
        self,
        model_directory: str | Path,
        *,
        device: str | None = None,
        max_length: int | None = None,
    ) -> None:
        directory = Path(model_directory)
        if (directory / "model").is_dir():
            directory = directory / "model"
        if not directory.is_dir():
            raise FileNotFoundError(directory)

        self.model = AutoModelForSequenceClassification.from_pretrained(
            directory, local_files_only=True,
        )
        expected = dict(enumerate(LABELS + ACTION_LABELS))
        if self.model.config.id2label != expected:
            raise ValueError("Model must contain category and action labels in training order")
        self.tokenizer = AutoTokenizer.from_pretrained(
            directory, use_fast=False, local_files_only=True,
        )
        position_limit = self.model.config.max_position_embeddings
        if max_length is None:
            config_path = directory.parent / "run_config.json"
            if config_path.is_file():
                max_length = json.loads(config_path.read_text(encoding="utf-8"))["max_length"]
            elif self.tokenizer.model_max_length <= position_limit:
                # Saved by training since the limit was recorded on the tokenizer.
                max_length = self.tokenizer.model_max_length
            else:
                raise ValueError("Supply max_length when run_config.json is unavailable")
        if type(max_length) is not int or max_length <= 0:
            raise ValueError("max_length must be a positive integer")
        if max_length > position_limit:
            raise ValueError("max_length exceeds the model's position limit")
        # Saved tokenizers (e.g. the ONNX bundle) then truncate correctly by default.
        self.tokenizer.model_max_length = max_length
        self.max_length = max_length
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model.to(self.device).eval()

    def predict(self, texts: list[str], batch_size: int = 32) -> list[Prediction]:
        """Return labels and independent task probabilities in input order.

        Action ``block_and_review`` means send for human review, independently
        of the predicted category. Texts are truncated to the configured token
        limit, including special tokens, and batched in order of token length.
        Empty input returns an empty list.
        """
        if not isinstance(texts, list) or any(not isinstance(text, str) for text in texts):
            raise TypeError("texts must be a list of strings")
        if type(batch_size) is not int or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        if not texts:
            return []
        encoded = self.tokenizer(texts, truncation=True, max_length=self.max_length)
        # Batch similar lengths together: most chat is ~10 tokens, and padding
        # every batch to a rare long message dominates the cost.
        order = sorted(range(len(texts)), key=lambda index: len(encoded["input_ids"][index]))
        results: list[Prediction | None] = [None] * len(texts)
        with torch.inference_mode():
            for start in range(0, len(order), batch_size):
                indices = order[start:start + batch_size]
                inputs = self.tokenizer.pad(
                    [{key: values[index] for key, values in encoded.items()} for index in indices],
                    padding=True, return_tensors="pt",
                ).to(self.device)
                logits = self.model(**inputs).logits.float()
                categories = logits[:, :len(LABELS)].softmax(dim=-1).cpu().tolist()
                actions = logits[:, len(LABELS):].softmax(dim=-1).cpu().tolist()
                for index, category, action in zip(indices, categories, actions, strict=True):
                    category_scores = dict(zip(LABELS, category, strict=True))
                    action_scores = dict(zip(ACTION_LABELS, action, strict=True))
                    results[index] = Prediction(
                        category=max(category_scores, key=category_scores.__getitem__),
                        action=max(action_scores, key=action_scores.__getitem__),
                        category_probabilities=category_scores,
                        action_probabilities=action_scores,
                    )
        return results
