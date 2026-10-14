"""DeBERTa student data preparation and soft-target distillation training."""

from .dataset import LABELS, prepare_dataset

__all__ = ["LABELS", "prepare_dataset"]
