"""Second-pass review dataset preparation."""

from pathlib import Path
from tempfile import TemporaryDirectory

import polars as pl


def generate_review_dataset(source: str | Path, destination: str | Path) -> Path:
    """Prepare second-pass native rows from category-scored chat JSONL.

    Accept prepared rows with ``state`` or legacy results with ``content``.
    Select the highest-probability category from the native answer and preserve
    the full first-pass record under ``source``. Ties use probability field
    order. Rows share one inferred schema, so fields missing from a row are
    written as null. Publish the destination atomically only after all rows
    are valid.
    """
    source, destination = Path(source), Path(destination)
    if source.resolve() == destination.resolve() or (
        destination.exists() and source.samefile(destination)
    ):
        raise ValueError("source and destination must differ")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(
        prefix="prepare-review-", dir=destination.parent
    ) as temporary:
        prepared = Path(temporary) / "prepared.jsonl"
        if source.stat().st_size == 0:
            prepared.touch()
        else:
            try:
                _ = _review_rows(source).sink_ndjson(prepared)
            except pl.exceptions.PolarsError as exc:
                raise ValueError(f"Invalid JSON rows in {source}") from exc
        prepared.replace(destination)
    return destination


def _review_rows(source: Path) -> pl.LazyFrame:
    records = pl.scan_ndjson(source)
    schema = records.collect_schema()
    if "state" in schema:
        state = pl.col("state")
    elif schema.get("content") == pl.String:
        state = pl.col("content")
    else:
        raise ValueError("Rows must have state or string content")

    dtype = schema.get("decision")
    for name in ("answer", "probabilities"):
        fields = {field.name: field.dtype for field in dtype.fields} if isinstance(dtype, pl.Struct) else {}
        dtype = fields.get(name)
    if not isinstance(dtype, pl.Struct) or not dtype.fields or any(
        not field.name or not field.dtype.is_numeric() for field in dtype.fields
    ):
        raise ValueError("Rows must have numeric category probabilities")
    labels = [field.name for field in dtype.fields]

    probabilities = pl.col("decision").struct.field("answer").struct.field("probabilities")
    scores = pl.concat_list(probabilities.struct.field(label) for label in labels)
    category = scores.list.arg_max().replace_strict(dict(enumerate(labels)))
    valid = (
        state.is_not_null()
        & scores.list.drop_nulls().list.len().gt(0)
        & scores.list.eval(pl.element().is_between(0, 1)).list.all()
    )
    invalid = (
        records.with_row_index("row", offset=1)
        .filter(~valid.fill_null(False))
        .select("row")
        .head(1)
        .collect()
    )
    if invalid.height:
        raise ValueError(
            f"Row {invalid.item()} must have a state and finite category probabilities in [0, 1]"
        )

    question = pl.struct(
        type=pl.lit("choice"),
        instructions=pl.format(
            "This message has been classified as {}. "
            "Consider it in the context of a video game involving combat, "
            "friendly insults, and fictional violence. The previous classification "
            "is a preliminary assessment, not a final decision.\n\n"
            "Is the message suitable to allow, or should it be blocked for human "
            "review? Distinguish ordinary gameplay discussion, fictional threats, "
            "and friendly banter from targeted abuse, hate speech, credible "
            "real-world threats, or other harmful content. Do not assume an insult "
            "is friendly solely because it occurs in a game.",
            category,
        ),
        criteria=pl.struct(
            allow=pl.lit("Suitable in this context; no review needed."),
            block_and_review=pl.lit(
                "Potentially harmful beyond ordinary gameplay or friendly banter; "
                "block for human review."
            ),
        ),
    )
    return records.select(
        pl.all().exclude("state", "question", "decision", "source", "content"),
        state.alias("state"),
        question.alias("question"),
        pl.struct(pl.all()).alias("source"),
    )
