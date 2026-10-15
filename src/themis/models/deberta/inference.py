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
    from ``run_config.json``. If that file was not downloaded, supply
    ``max_length`` explicitly using the value used during training.
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
        if max_length is None:
            config_path = directory.parent / "run_config.json"
            if not config_path.is_file():
                raise ValueError("Supply max_length when run_config.json is unavailable")
            max_length = json.loads(config_path.read_text(encoding="utf-8"))["max_length"]
        if type(max_length) is not int or max_length <= 0:
            raise ValueError("max_length must be a positive integer")

        self.model = AutoModelForSequenceClassification.from_pretrained(
            directory, local_files_only=True,
        )
        expected = dict(enumerate(LABELS + ACTION_LABELS))
        if self.model.config.id2label != expected:
            raise ValueError("Model must contain category and action labels in training order")
        if max_length > self.model.config.max_position_embeddings:
            raise ValueError("max_length exceeds the model's position limit")
        self.tokenizer = AutoTokenizer.from_pretrained(
            directory, use_fast=False, local_files_only=True,
        )
        self.max_length = max_length
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model.to(self.device).eval()

    def predict(self, texts: list[str], batch_size: int = 32) -> list[Prediction]:
        """Return labels and independent task probabilities in input order.

        Action ``block_and_review`` means send for human review, independently
        of the predicted category. Texts are truncated to the configured token
        limit, including special tokens. Empty input returns an empty list.
        """
        if not isinstance(texts, list) or any(not isinstance(text, str) for text in texts):
            raise TypeError("texts must be a list of strings")
        if type(batch_size) is not int or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        results: list[Prediction] = []
        with torch.inference_mode():
            for start in range(0, len(texts), batch_size):
                inputs = self.tokenizer(
                    texts[start:start + batch_size], padding=True, truncation=True,
                    max_length=self.max_length, return_tensors="pt",
                ).to(self.device)
                logits = self.model(**inputs).logits.float()
                categories = logits[:, :len(LABELS)].softmax(dim=-1).cpu().tolist()
                actions = logits[:, len(LABELS):].softmax(dim=-1).cpu().tolist()
                for category, action in zip(categories, actions, strict=True):
                    category_scores = dict(zip(LABELS, category, strict=True))
                    action_scores = dict(zip(ACTION_LABELS, action, strict=True))
                    results.append(Prediction(
                        category=max(category_scores, key=category_scores.__getitem__),
                        action=max(action_scores, key=action_scores.__getitem__),
                        category_probabilities=category_scores,
                        action_probabilities=action_scores,
                    ))
        return results
