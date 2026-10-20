from pathlib import Path

import polars as pl

from themis.datasets.deduplication import deduplicate_chats
from themis.datasets.download_2b2t_chat import download as download_2b2t_chats
from themis.datasets.download_minecraft_server_chat import (
    download as download_minecraft_chats,
)
from themis.datasets.review import generate_review_dataset
from themis.datasets.sampling import guided_sample_dataset
from themis.models.deberta.workflow import (
    compare_correction_locally,
    correct_student_locally,
    export_if_missing,
    train_student_locally,
)
from themis.models.decision.perplexity.bulk import predict_dataset
from themis.models.decision.perplexity.dataset import prepare_dataset
from themis.models.decision.perplexity.questions import CATEGORY_QUESTION
from themis.models.koala.moderation import moderate_dataset


def pipeline():
    """Score 100k chats for category and review, train and post-train the DeBERTa student, then compare them."""
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
    # Train locally from scratch: the student is now mDeBERTa, not DeBERTa-v3.
    model_directory = train_student_locally(
        source=review_decisions,
        training_data_directory="cache/deberta/review-splits",
        output_directory="cache/deberta/local-run",
        config={"bf16": True},
    )
    _ = export_if_missing(model_directory)
    # Post-train the distilled student on reviewed labels the teacher got wrong.
    corrected_directory = correct_student_locally(
        labeled_path=Path("moderation_audit/all_samples.jsonl"),
        model_directory=model_directory,
        teacher_split_directory="cache/deberta/review-splits",
        output_directory="cache/deberta/local-correction-run",
        config={"bf16": True},
    )
    _ = export_if_missing(corrected_directory)
    # Does post-training help on held-out reviewed labels without forgetting the teacher?
    _ = compare_correction_locally(
        model_directory=model_directory,
        corrected_directory=corrected_directory,
        correction_data_directory="cache/deberta/correction-splits",
        config={"bf16": True},
    )
    print("Pipeline complete.", flush=True)


def main() -> None:
    pipeline()


if __name__ == "__main__":
    main()
