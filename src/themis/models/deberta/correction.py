"""Prepare post-training splits from reviewed category labels.

The distilled student only imitates the teacher, so it inherits the teacher's
blind spots (for example obfuscated slurs). Post-training continues from the
distilled checkpoint on reviewed labels, mixed with replayed teacher rows so
the student does not forget the rest of the distribution.
"""

import json
import unicodedata
from pathlib import Path

import polars as pl

from .dataset import ACTION_LABELS, LABELS, assign_grouped_splits

# Reviewed labels carry a category only; these categories imply the action.
# Profanity and insult depend on context, so rows with them must give "action".
DEFAULT_ACTIONS = {"none": "allow", "slur": "block_and_review",
                   "sexual": "block_and_review", "threat": "block_and_review"}

# "!" and "|" are left out: as punctuation they would split "word!" from "word".
_LOOKALIKES = str.maketrans("0134578@$", "oieastbas")


def obfuscation_key(text: str) -> str:
    """Collapse case, lookalike digits/symbols, spacing and punctuation.

    "Go h0me n0w" and "go h.o.m.e now" share a key, so a
    message's obfuscated variants never straddle train and test.
    """
    folded = unicodedata.normalize("NFKC", text).casefold().translate(_LOOKALIKES)
    return "".join(char for char in folded if char.isalnum())


def _smoothed(index: int, size: int, smoothing: float) -> list[float]:
    return [1 - smoothing + smoothing / size if i == index else smoothing / size for i in range(size)]


def prepare_correction_dataset(
    labeled_path: str | Path,
    teacher_split_directory: str | Path,
    output_directory: str | Path,
    *,
    replay_ratio: float = 2.0,
    label_smoothing: float = 0.05,
    seed: int = 42,
) -> dict[str, Path]:
    """Write post-training splits in the trainer's Parquet format.

    labeled_path is JSONL with ``text`` and ``category`` (and optionally
    ``action``). Reviewed rows are split approximately 80/10/10, keeping rows
    with the same obfuscation_key together. Their hard labels become smoothed
    targets, since one-hot targets push logits without bound.

    train and validation add replay_ratio teacher rows per reviewed row, drawn
    from the matching teacher split. test holds reviewed rows only, measuring
    the target corrections; teacher_test is the unchanged teacher test split,
    measuring regressions. Use a new output directory.
    """
    if not 0 <= label_smoothing < 1:
        raise ValueError("label_smoothing must be in [0, 1)")
    if replay_ratio < 0:
        raise ValueError("replay_ratio must be nonnegative")
    teacher = Path(teacher_split_directory)
    for split in ("train", "validation", "test"):
        if not (teacher / f"{split}.parquet").is_file():
            raise FileNotFoundError(teacher / f"{split}.parquet")

    rows, seen = [], {}  # seen: text -> (category, action)
    with Path(labeled_path).open(encoding="utf-8") as reader:
        for number, line in enumerate(reader, 1):
            try:
                record = json.loads(line)
                text, category = record["text"], record["category"]
                action = record.get("action", DEFAULT_ACTIONS.get(category))
                if not isinstance(text, str) or not text.strip():
                    raise ValueError("Expected nonempty string text")
                if category not in LABELS:
                    raise ValueError(f"Unknown category {category!r}")
                if action not in ACTION_LABELS:
                    raise ValueError(f"Missing or unknown action for category {category!r}")
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"Invalid labeled sample on row {number}: {exc}") from exc
            if text in seen:
                if seen[text] != (category, action):
                    raise ValueError(f"Conflicting labels for identical text on row {number}")
                continue
            seen[text] = (category, action)
            category_index, action_index = LABELS.index(category), ACTION_LABELS.index(action)
            rows.append({
                "content": text,
                "targets": _smoothed(category_index, len(LABELS), label_smoothing),
                "teacher_argmax": category_index,
                "action_targets": _smoothed(action_index, len(ACTION_LABELS), label_smoothing),
                "action_teacher_argmax": action_index,
                "group": obfuscation_key(text) or text,
            })
    if not rows:
        raise ValueError("No labeled samples")
    columns = ("content", "targets", "teacher_argmax", "action_targets", "action_teacher_argmax")
    reviewed = assign_grouped_splits(
        pl.DataFrame(rows).with_columns(
            pl.col("targets").cast(pl.Array(pl.Float32, len(LABELS))),
            pl.col("teacher_argmax").cast(pl.UInt8),
            pl.col("action_targets").cast(pl.Array(pl.Float32, len(ACTION_LABELS))),
            pl.col("action_teacher_argmax").cast(pl.UInt8),
        ),
        group_column="group", class_column="teacher_argmax", seed=seed,
    )

    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=False)
    paths, counts = {}, {}
    for split in ("train", "validation", "test"):
        part = reviewed.filter(pl.col("split") == split).select(columns)
        counts[split] = {"reviewed": part.height, "teacher": 0}
        if split != "test":
            replay = pl.read_parquet(teacher / f"{split}.parquet", columns=list(columns))
            replay = replay.sample(min(replay.height, round(replay_ratio * part.height)), seed=seed)
            counts[split]["teacher"] = replay.height
            part = pl.concat([part, replay.cast(part.schema)])
        paths[split] = output_directory / f"{split}.parquet"
        part.write_parquet(paths[split])
    paths["teacher_test"] = output_directory / "teacher_test.parquet"
    teacher_test = pl.read_parquet(teacher / "test.parquet", columns=list(columns))
    teacher_test.write_parquet(paths["teacher_test"])
    counts["teacher_test"] = {"reviewed": 0, "teacher": teacher_test.height}
    (output_directory / "manifest.json").write_text(json.dumps({
        "labeled_path": str(labeled_path), "teacher_split_directory": str(teacher),
        "replay_ratio": replay_ratio, "label_smoothing": label_smoothing, "seed": seed, "counts": counts,
    }, indent=2) + "\n", encoding="utf-8")
    return paths
