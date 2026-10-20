"""Create train/validation/test Parquet files from the local teacher dataset."""

import json
from collections.abc import Mapping
from math import isfinite
from pathlib import Path
from typing import Any

import polars as pl

LABELS = ("none", "profanity", "insult", "slur", "sexual", "threat")
ACTION_LABELS = ("allow", "block_and_review")


def _targets(record: Mapping[str, Any], labels: tuple[str, ...]) -> list[float]:
    probabilities = record["decision"]["answer"]["probabilities"]
    if not isinstance(probabilities, dict) or set(probabilities) != set(labels):
        raise ValueError(f"Expected probabilities for {labels}")
    values = [probabilities[label] for label in labels]
    if any(isinstance(value, bool) or not isinstance(value, (int, float))
           or not isfinite(value) or not 0 <= value <= 1 for value in values) or sum(values) <= 0:
        raise ValueError("Expected finite nonnegative probabilities with positive mass")
    return [value / sum(values) for value in values]


def assign_grouped_splits(frame: pl.DataFrame, *, group_column: str, class_column: str,
                          seed: int = 42) -> pl.DataFrame:
    """Add an approximately 80/10/10 ``split`` column, keeping each group together.

    Groups are stratified by their modal class; cumulative row counts handle
    groups of different sizes.
    """
    split_groups = frame.group_by(group_column).agg(
        pl.col(class_column).mode().min().alias("group_class"), pl.len().alias("group_size"),
    ).with_columns(pl.col(group_column).hash(seed=seed).alias("shuffle_key"))
    split_groups = split_groups.sort("group_class", "shuffle_key", group_column).with_columns(
        ((pl.col("group_size").cum_sum().over("group_class") - pl.col("group_size") / 2)
         / pl.col("group_size").sum().over("group_class")).alias("split_position"),
    ).with_columns(
        pl.when(pl.col("split_position") < 0.8).then(pl.lit("train"))
        .when(pl.col("split_position") < 0.9).then(pl.lit("validation"))
        .otherwise(pl.lit("test")).alias("split"),
    )
    return frame.join(split_groups.select(group_column, "split"), on=group_column, maintain_order="left")


def prepare_dataset(source_path: str | Path, output_directory: str | Path, *, seed: int = 42) -> dict[str, Path]:
    """Write approximately 80/10/10 splits, keeping normalized variants together.

    Input is the pipeline's review_decisions JSONL: category scores are under
    source.decision and action scores under decision. Each file contains both
    distributions and their argmax labels. Use a new
    output directory; no existing files are overwritten. Return paths keyed by
    train, validation, and test.
    """
    rows = []
    with Path(source_path).open(encoding="utf-8") as reader:
        for number, line in enumerate(reader, 1):
            try:
                record = json.loads(line)
                content = record["state"] if "state" in record else record["content"]
                if not isinstance(content, str):
                    raise ValueError("Expected string chat state")
                targets = _targets(record["source"], LABELS)
                action_targets = _targets(record, ACTION_LABELS)
                if content.strip():
                    rows.append({"content": content, "targets": targets, "action_targets": action_targets})
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"Invalid review result on row {number}: {exc}") from exc
    if not rows:
        raise ValueError("No nonempty chats in review results")
    prepared_messages = pl.DataFrame(rows).with_columns(
        pl.col("targets").list.arg_max().cast(pl.UInt8).alias("teacher_argmax"),
        pl.col("action_targets").list.arg_max().cast(pl.UInt8).alias("action_teacher_argmax"),
        pl.col("content").map_elements(lambda text: " ".join(text.casefold().split()), return_dtype=pl.String)
        .alias("normalized_content"),
    ).with_columns(
        pl.col("targets").cast(pl.Array(pl.Float32, len(LABELS))),
        pl.col("action_targets").cast(pl.Array(pl.Float32, len(ACTION_LABELS))),
    )

    prepared_messages = assign_grouped_splits(
        prepared_messages, group_column="normalized_content", class_column="teacher_argmax", seed=seed,
    )
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=False)
    split_paths = {split_name: output_directory / f"{split_name}.parquet"
                   for split_name in ("train", "validation", "test")}
    for split_name, file_path in split_paths.items():
        prepared_messages.filter(pl.col("split") == split_name).select(
            "content", "targets", "teacher_argmax", "action_targets", "action_teacher_argmax",
        ).write_parquet(file_path)
    return split_paths
