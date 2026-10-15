import json
from math import isfinite
from pathlib import Path
from tempfile import TemporaryDirectory

import polars as pl

from distill_jev.datasets.deduplication import deduplicate_chats
from distill_jev.datasets.download_2b2t_chat import download as download_2b2t_chats
from distill_jev.datasets.download_minecraft_server_chat import (
    download as download_minecraft_chats,
)
from distill_jev.datasets.sampling import guided_sample_dataset
from distill_jev.models.deberta.dataset import prepare_dataset as prepare_training_dataset
from distill_jev.models.decision.perplexity.bulk import predict_dataset
from distill_jev.models.decision.perplexity.dataset import prepare_dataset
from distill_jev.models.decision.perplexity.questions import CATEGORY_QUESTION
from distill_jev.models.koala.moderation import moderate_dataset


def pipeline() -> None:
    """Score 100k chats for category and review, then train the DeBERTa student."""
    deduplicated = Path("cache/combined_chats_deduplicated.jsonl")
    koala_predictions = Path("cache/combined_chats_moderated.jsonl")
    sample = Path("cache/combined_chats_inference_sample_100k.jsonl")
    prepared = Path("cache/combined_chats_inference_sample_100k_decision_inputs.jsonl")
    decisions = Path("cache/combined_chats_inference_sample_100k_decisions.jsonl")
    review_inputs = Path("cache/combined_chats_inference_sample_100k_review_inputs.jsonl")
    review_decisions = Path("cache/combined_chats_inference_sample_100k_review_decisions.jsonl")

    # Moderation publishes this file only after all result shards are downloaded.
    if koala_predictions.is_file():
        print(f"Reusing completed results: {koala_predictions}", flush=True)
    else:
        # Preserve input bytes/order so failed jobs can reuse their checkpoints.
        if not deduplicated.exists():
            chats = pl.concat(
                items=[
                    download_minecraft_chats(
                        output_path=Path("cache/minecraft_chats.jsonl"), force=False
                    ),
                    download_2b2t_chats(
                        output_path=Path("cache/2b2t_chats.csv"), force=False
                    ),
                ]
            )
            _ = deduplicate_chats(source=chats, destination=deduplicated)

        # Matching checkpoints are resumed after a failed moderation run.
        _ = moderate_dataset(
            input_path=deduplicated,
            output_path=koala_predictions,
            text_field="content",
            batch_size=128,
        )
    _ = guided_sample_dataset(
        source=koala_predictions,
        destination=sample,
        quotas={
            "OK": 49_911,
            "H": 20_000,
            "V": 10_000,
            "S": 8_000,
            "SH": 7_000,
            "HR": 5_088,
            "H2": 1,
        },
        seed=42,
    )
    if decisions.is_file():
        print(f"Reusing completed results: {decisions}", flush=True)
    else:
        _ = prepare_dataset(
            source=sample, destination=prepared, question=CATEGORY_QUESTION
        )
        _ = predict_dataset(
            input_path=prepared,
            output_path=decisions,
            batch_size=32,
            rows_per_shard=1024,
            max_containers=4,
        )
    if review_decisions.is_file():
        print(f"Reusing completed results: {review_decisions}", flush=True)
    else:
        _ = generate_review_dataset(source=decisions, destination=review_inputs)
        _ = predict_dataset(
            input_path=review_inputs,
            output_path=review_decisions,
            batch_size=32,
            rows_per_shard=1024,
            max_containers=4,
        )
    train_student(review_decisions)
    print("Pipeline complete.", flush=True)


def export_student(
    model_directory: str | Path,
    output_directory: str | Path = "cache/deberta/onnx",
    *,
    max_length: int | None = None,
) -> Path:
    """Export a local/downloaded training run; requires training and onnx extras."""
    from distill_jev.models.deberta.onnx_export import export_onnx

    return export_onnx(model_directory, output_directory, max_length=max_length)


def train_student(source: str | Path) -> None:
    """Prepare dual-target splits and train on Modal, recording completed runs."""
    splits = Path("cache/deberta/review-splits")
    # output = Path("cache/deberta/review-run")
    receipt = splits.parent / "review-modal-run.json"
    if receipt.is_file():
        print(f"Reusing completed Modal training: {receipt}", flush=True)
        return
    # Local training alternative:
    # if (output / "metrics.json").is_file() and (output / "model" / "config.json").is_file():
    #     print(f"Reusing completed training: {output}", flush=True)
    #     return
    # if output.exists():
    #     raise FileExistsError(f"Incomplete training output at {output}; move it aside before retrying")

    from distill_jev.models.deberta.modal_training import train

    # from distill_jev.models.deberta.training import train

    if all((splits / f"{split}.parquet").is_file() for split in ("train", "validation", "test")):
        print(f"Reusing training splits: {splits}", flush=True)
    else:
        prepare_training_dataset(source, splits)
    result = train(splits)
    temporary = receipt.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    temporary.replace(receipt)
    # train(splits, output)


def generate_review_dataset(source: str | Path, destination: str | Path) -> Path:
    """Prepare second-pass native rows from category-scored chat JSONL.

    Accept prepared rows with ``state`` or legacy results with ``content``.
    Select the highest-probability category from the native answer and preserve
    the full first-pass record under ``source``. Ties use probability insertion
    order. Stream rows without altering metadata or running additional inference;
    publish the destination atomically only after all rows are valid.
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
        with source.open("rb") as reader, prepared.open("wb") as writer:
            for number, line in enumerate(reader, 1):
                try:
                    record = json.loads(line)
                except (ValueError, UnicodeDecodeError) as exc:
                    raise ValueError(f"Invalid JSON on row {number}") from exc
                if not isinstance(record, dict):
                    raise ValueError(f"Row {number} must be an object")
                if "state" in record:
                    state = record["state"]
                    if state is None:
                        raise ValueError(f"Row {number} must have a non-null state")
                else:
                    state = record.get("content")
                    if not isinstance(state, str):
                        raise ValueError(f"Row {number} must have state or string content")
                try:
                    probabilities = record["decision"]["answer"]["probabilities"]
                except (KeyError, TypeError) as exc:
                    raise ValueError(f"Row {number} must have category probabilities") from exc
                if not isinstance(probabilities, dict) or not probabilities or any(
                    not isinstance(label, str) or not label
                    or isinstance(score, bool) or not isinstance(score, (int, float))
                    or not 0 <= score <= 1 or not isfinite(score)
                    for label, score in probabilities.items()
                ):
                    raise ValueError(f"Row {number} must have finite category probabilities in [0, 1]")
                category = max(probabilities, key=probabilities.__getitem__)
                question = {
                    "type": "choice",
                    "instructions": (
                        f"This message has been classified as {category}. "
                        "Consider it in the context of a video game involving combat, "
                        "friendly insults, and fictional violence. The previous classification "
                        "is a preliminary assessment, not a final decision.\n\n"
                        "Is the message suitable to allow, or should it be blocked for human "
                        "review? Distinguish ordinary gameplay discussion, fictional threats, "
                        "and friendly banter from targeted abuse, hate speech, credible "
                        "real-world threats, or other harmful content. Do not assume an insult "
                        "is friendly solely because it occurs in a game."
                    ),
                    "criteria": {
                        "allow": "Suitable in this context; no review needed.",
                        "block_and_review": (
                            "Potentially harmful beyond ordinary gameplay or friendly banter; "
                            "block for human review."
                        ),
                    },
                }
                row = {
                    name: value for name, value in record.items()
                    if name not in {"state", "question", "decision", "source", "content"}
                }
                row.update(state=state, question=question, source=record)
                try:
                    payload = (json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"Invalid JSON on row {number}") from exc
                writer.write(payload)
        prepared.replace(destination)
    return destination


def main() -> None:
    pipeline()


if __name__ == "__main__":
    main()
