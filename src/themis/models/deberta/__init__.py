"""DeBERTa student data preparation and soft-target distillation training."""

from .dataset import ACTION_LABELS, LABELS, prepare_dataset

# Multilingual DeBERTa-v3 (CC100, 250k-piece vocabulary), so chat in other
# languages tokenizes into words rather than byte fallback.
BASE_MODEL = "microsoft/mdeberta-v3-base"

__all__ = ["ACTION_LABELS", "BASE_MODEL", "LABELS", "prepare_dataset"]
