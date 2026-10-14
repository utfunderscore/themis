"""Create train/validation/test Parquet files from the local teacher dataset."""

from pathlib import Path

import polars as pl

LABELS = ("none", "profanity", "insult", "slur", "sexual", "threat")


def prepare_dataset(source_path: str | Path, output_directory: str | Path, *, seed: int = 42) -> dict[str, Path]:
    """Write approximately 80/10/10 splits, keeping normalized variants together.

    Each file contains original content, six float32 targets in LABELS order, and
    teacher_argmax. Input is the already-inspected local teacher JSONL. Use a new
    output directory; no existing files are overwritten. Return paths keyed by
    train, validation, and test.
    """
    teacher_probabilities = pl.col("decision").struct.field("answer").struct.field("probabilities")
    prepared_messages = pl.read_ndjson(source_path, infer_schema_length=None).filter(
        pl.col("content").str.strip_chars() != "",
    ).select(
        "content",
        pl.concat_list([teacher_probabilities.struct.field(label) for label in LABELS]).alias("targets"),
    ).with_columns(
        pl.col("targets").list.arg_max().cast(pl.UInt8).alias("teacher_argmax"),
        pl.col("content").map_elements(lambda text: " ".join(text.casefold().split()), return_dtype=pl.String)
        .alias("normalized_content"),
    ).with_columns(
        pl.concat_list([pl.col("targets").list.get(i) / pl.col("targets").list.sum() for i in range(6)])
        .cast(pl.Array(pl.Float32, 6)).alias("targets"),
    )

    # Stratify by each group's modal class; cumulative row counts handle variants.
    split_groups = prepared_messages.group_by("normalized_content").agg(
        pl.col("teacher_argmax").mode().min().alias("teacher_class"), pl.len().alias("group_size"),
    ).with_columns(pl.col("normalized_content").hash(seed=seed).alias("shuffle_key"))
    split_groups = split_groups.sort("teacher_class", "shuffle_key", "normalized_content").with_columns(
        ((pl.col("group_size").cum_sum().over("teacher_class") - pl.col("group_size") / 2)
         / pl.col("group_size").sum().over("teacher_class")).alias("split_position"),
    ).with_columns(
        pl.when(pl.col("split_position") < 0.8).then(pl.lit("train"))
        .when(pl.col("split_position") < 0.9).then(pl.lit("validation"))
        .otherwise(pl.lit("test")).alias("split"),
    )
    prepared_messages = prepared_messages.join(
        split_groups.select("normalized_content", "split"), on="normalized_content", maintain_order="left",
    )
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=False)
    split_paths = {split_name: output_directory / f"{split_name}.parquet"
                   for split_name in ("train", "validation", "test")}
    for split_name, file_path in split_paths.items():
        prepared_messages.filter(pl.col("split") == split_name).select(
            "content", "targets", "teacher_argmax",
        ).write_parquet(file_path)
    return split_paths
