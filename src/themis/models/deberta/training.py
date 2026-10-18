"""Distill category and moderation action. Requires the ``training`` extra."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import polars as pl
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    Trainer,
    TrainingArguments,
    set_seed,
)

from . import BASE_MODEL
from .dataset import ACTION_LABELS, LABELS


@dataclass(frozen=True)
class TrainingConfig:
    model_name: str = BASE_MODEL
    revision: str = "main"
    max_length: int = 256
    epochs: float = 3.0
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    train_batch_size: int = 16
    eval_batch_size: int = 32
    gradient_accumulation_steps: int = 1
    seed: int = 42
    bf16: bool = False
    fp16: bool = False
    gradient_checkpointing: bool = False
    use_cpu: bool = False

    def __post_init__(self) -> None:
        for name in ("max_length", "train_batch_size", "eval_batch_size", "gradient_accumulation_steps"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("epochs", "learning_rate"):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not np.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("weight_decay must be finite and nonnegative")
        if not 0 <= self.warmup_ratio <= 1:
            raise ValueError("warmup_ratio must be between zero and one")
        if self.bf16 and self.fp16:
            raise ValueError("Choose only one of bf16 and fp16")
        if self.use_cpu and self.fp16:
            raise ValueError("fp16 requires a supported accelerator")


class SoftTargetDataset(Dataset):
    """Validate prepared Parquet and tokenize once, leaving padding to batches."""

    def __init__(self, path: str | Path, tokenizer, max_length: int) -> None:
        frame = pl.read_parquet(path, columns=["content", "targets", "teacher_argmax"])
        if frame.is_empty():
            raise ValueError(f"{path}: split must not be empty")
        texts = frame["content"].to_list()
        if any(not isinstance(text, str) or not text.strip() for text in texts):
            raise ValueError(f"{path}: content must contain nonempty strings")
        try:
            targets = np.asarray(frame["targets"].to_list(), dtype=np.float32)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{path}: invalid targets") from exc
        if targets.shape != (len(texts), len(LABELS)):
            raise ValueError(f"{path}: expected six targets per row in LABELS order")
        if (not np.isfinite(targets).all() or (targets < 0).any()
                or not np.allclose(targets.sum(axis=1), 1.0, atol=1e-5, rtol=0)):
            raise ValueError(f"{path}: targets must be finite, nonnegative probability distributions")
        if not np.array_equal(targets.argmax(axis=1), frame["teacher_argmax"].to_numpy()):
            raise ValueError(f"{path}: teacher_argmax disagrees with targets")
        actions = pl.read_parquet(path, columns=["action_targets", "action_teacher_argmax"])
        try:
            action_targets = np.asarray(actions["action_targets"].to_list(), dtype=np.float32)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{path}: invalid action targets") from exc
        if (action_targets.shape != (len(texts), len(ACTION_LABELS))
                or not np.isfinite(action_targets).all() or (action_targets < 0).any()
                or not np.allclose(action_targets.sum(axis=1), 1.0, atol=1e-5, rtol=0)):
            raise ValueError(f"{path}: action targets must be probability distributions")
        if not np.array_equal(action_targets.argmax(axis=1), actions["action_teacher_argmax"].to_numpy()):
            raise ValueError(f"{path}: action_teacher_argmax disagrees with targets")
        self.targets = np.concatenate((targets, action_targets), axis=1)
        self.encodings = {}
        for start in range(0, len(texts), 1024):
            batch = tokenizer(texts[start:start + 1024], truncation=True, max_length=max_length)
            for key, values in batch.items():
                self.encodings.setdefault(key, []).extend(values)

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int) -> dict:
        return {**{key: values[index] for key, values in self.encodings.items()},
                "labels": self.targets[index].tolist()}


def soft_target_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Categorical cross-entropy with teacher probabilities, computed in FP32.

    This has the same gradients as KL(teacher || student). These six labels
    form one distribution, so sigmoid/BCE and hard argmax training are incorrect.
    """
    return -(targets.float() * logits.float().log_softmax(dim=-1)).sum(dim=-1).mean()


class SoftTargetTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Loss is a per-microbatch mean, not normalized by Trainer's item count.
        # Trainer must apply gradient-accumulation scaling itself.
        self.model_accepts_loss_kwargs = False

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        targets = inputs["labels"]
        outputs = model(**{key: value for key, value in inputs.items() if key != "labels"})
        loss = multitask_loss(outputs.logits, targets)
        return (loss, outputs) if return_outputs else loss


def multitask_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Equal-weight sum of independently normalized category and action losses."""
    boundary = len(LABELS)
    return (soft_target_loss(logits[..., :boundary], targets[..., :boundary])
            + soft_target_loss(logits[..., boundary:], targets[..., boundary:]))


def compute_metrics(prediction: EvalPrediction) -> dict[str, float]:
    metrics = {}
    for name, start, stop, labels in (
        ("category", 0, len(LABELS), LABELS),
        ("action", len(LABELS), len(LABELS) + len(ACTION_LABELS), ACTION_LABELS),
    ):
        task = EvalPrediction(predictions=prediction.predictions[:, start:stop],
                              label_ids=prediction.label_ids[:, start:stop])
        metrics.update({f"{name}_{key}": value for key, value in _task_metrics(task, labels).items()})
    return metrics


def _task_metrics(prediction: EvalPrediction, labels: tuple[str, ...]) -> dict[str, float]:
    """Agreement metrics measure fidelity to the teacher, not human accuracy."""
    logits = np.asarray(prediction.predictions, dtype=np.float64)
    targets = np.asarray(prediction.label_ids, dtype=np.float64)
    shifted = logits - logits.max(axis=1, keepdims=True)
    log_probs = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    predicted = logits.argmax(axis=1)
    expected = targets.argmax(axis=1)
    metrics = {"teacher_agreement": float(np.mean(predicted == expected))}
    f1_scores = []
    for index, label in enumerate(labels):
        true_positive = np.sum((predicted == index) & (expected == index))
        denominator = np.sum(predicted == index) + np.sum(expected == index)
        f1 = float(2 * true_positive / denominator) if denominator else 0.0
        metrics[f"teacher_f1_{label}"] = f1
        f1_scores.append(f1)
    metrics["teacher_macro_f1"] = float(np.mean(f1_scores))
    metrics["teacher_kl"] = float(np.mean(np.sum(
        targets * (np.log(np.maximum(targets, np.finfo(np.float64).tiny)) - log_probs), axis=1,
    )))
    return metrics


def train(
    split_directory: str | Path,
    output_directory: str | Path,
    *,
    config: TrainingConfig | None = None,
) -> dict[str, object]:
    """Train on train.parquet, select by validation loss, then evaluate test once.

    Any other Parquet files in split_directory are also evaluated once, with
    metrics keyed by file stem. config.model_name may be a local checkpoint
    with the same eight outputs, which continues training it (post-training).
    Run on the current machine (CPU or GPU). Output must be a new directory.
    Returns artifact paths and metrics; checkpoints retain optimizer state.
    """
    config = config or TrainingConfig()
    source, destination = Path(split_directory), Path(output_directory)
    if destination.exists():
        raise FileExistsError(destination)
    for split in ("train", "validation", "test"):
        if not (source / f"{split}.parquet").is_file():
            raise FileNotFoundError(source / f"{split}.parquet")
    set_seed(config.seed)
    # Preserve SentencePiece byte fallback for unusual chat characters.
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name, revision=config.revision, use_fast=False,
    )
    # The base tokenizer has no limit; save the trained one with ours.
    tokenizer.model_max_length = config.max_length
    extra_splits = sorted(path.stem for path in source.glob("*.parquet")
                          if path.stem not in ("train", "validation", "test"))
    datasets = {split: SoftTargetDataset(source / f"{split}.parquet", tokenizer, config.max_length)
                for split in ("train", "validation", "test", *extra_splits)}
    model = AutoModelForSequenceClassification.from_pretrained(
        config.model_name, revision=config.revision, num_labels=len(LABELS) + len(ACTION_LABELS),
        id2label=dict(enumerate(LABELS + ACTION_LABELS)),
        label2id={label: i for i, label in enumerate(LABELS + ACTION_LABELS)},
    )
    model.config.task_label_groups = {"category": list(LABELS), "action": list(ACTION_LABELS)}
    if config.max_length > model.config.max_position_embeddings:
        raise ValueError("max_length exceeds the model's position limit")
    destination.mkdir(parents=True, exist_ok=False)
    (destination / "run_config.json").write_text(json.dumps({
        **asdict(config), "labels": LABELS, "action_labels": ACTION_LABELS,
        "loss": "category_cross_entropy + action_cross_entropy", "split_directory": str(source.resolve()),
        "resolved_model_revision": getattr(model.config, "_commit_hash", None),
    }, indent=2) + "\n", encoding="utf-8")
    arguments = TrainingArguments(
        output_dir=str(destination / "checkpoints"),
        num_train_epochs=config.epochs,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        per_device_train_batch_size=config.train_batch_size,
        per_device_eval_batch_size=config.eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        gradient_checkpointing=config.gradient_checkpointing,
        bf16=config.bf16, fp16=config.fp16, use_cpu=config.use_cpu,
        seed=config.seed, data_seed=config.seed,
        eval_strategy="epoch", save_strategy="epoch", save_total_limit=2,
        load_best_model_at_end=True, metric_for_best_model="eval_loss", greater_is_better=False,
        logging_steps=50, report_to="none", optim="adamw_torch",
        label_names=["labels"], eval_accumulation_steps=16,
    )
    trainer = SoftTargetTrainer(
        model=model, args=arguments, processing_class=tokenizer,
        train_dataset=datasets["train"], eval_dataset=datasets["validation"],
        data_collator=DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8),
        compute_metrics=compute_metrics,
    )
    train_result = trainer.train()
    validation = trainer.evaluate(metric_key_prefix="validation")
    test = trainer.evaluate(datasets["test"], metric_key_prefix="test")
    extra = {split: trainer.evaluate(datasets[split], metric_key_prefix=split) for split in extra_splits}
    model_directory = destination / "model"
    trainer.save_model(str(model_directory))
    tokenizer.save_pretrained(model_directory)
    metrics = {"train": train_result.metrics, "validation": validation, "test": test,
               **extra, "best_checkpoint": trainer.state.best_model_checkpoint}
    (destination / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return {"model_directory": model_directory, "metrics": metrics}


def evaluate(
    model_directory: str | Path,
    split_paths: dict[str, str | Path],
    *,
    config: TrainingConfig | None = None,
) -> dict[str, dict[str, float]]:
    """Evaluate a trained model on prepared Parquet splits without training it.

    Returns metrics keyed by split name, prefixed like a run's metrics.json.
    config supplies max_length, eval_batch_size and precision; model_name is ignored.
    """
    config = config or TrainingConfig()
    tokenizer = AutoTokenizer.from_pretrained(model_directory, use_fast=False)
    model = AutoModelForSequenceClassification.from_pretrained(model_directory)
    with TemporaryDirectory(prefix="deberta-evaluate-") as scratch:
        trainer = SoftTargetTrainer(
            model=model, processing_class=tokenizer,
            args=TrainingArguments(
                output_dir=scratch, per_device_eval_batch_size=config.eval_batch_size,
                bf16=config.bf16, fp16=config.fp16, use_cpu=config.use_cpu, report_to="none",
                label_names=["labels"], eval_accumulation_steps=16,
            ),
            data_collator=DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8),
            compute_metrics=compute_metrics,
        )
        return {split: trainer.evaluate(SoftTargetDataset(path, tokenizer, config.max_length),
                                        metric_key_prefix=split)
                for split, path in split_paths.items()}
