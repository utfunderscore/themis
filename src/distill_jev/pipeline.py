from pathlib import Path

import polars as pl

from distill_jev.datasets.deduplication import deduplicate_chats
from distill_jev.datasets.download_2b2t_chat import download as download_2b2t_chats
from distill_jev.datasets.download_minecraft_server_chat import (
    download as download_minecraft_chats,
)
from distill_jev.datasets.sampling import guided_sample_dataset
from distill_jev.models.koala.moderation import moderate_dataset

def pipeline() -> None:
    """Deduplicate, reuse/resume Koala moderation, then build the guided 100k sample."""
    deduplicated = Path("cache/combined_chats_deduplicated.jsonl")
    koala_predictions = Path("cache/combined_chats_moderated.jsonl")
    sample = Path("cache/combined_chats_inference_sample_100k.jsonl")

    # Moderation publishes this file only after all result shards are downloaded.
    if koala_predictions.is_file():
        print(f"Reusing completed results: {koala_predictions}", flush=True)
    else:
        # Preserve input bytes/order so failed jobs can reuse their checkpoints.
        if not deduplicated.exists():
            chats = pl.concat([
                download_minecraft_chats(Path("cache/minecraft_chats.jsonl"), force=False),
                download_2b2t_chats(Path("cache/2b2t_chats.csv"), force=False),
            ])
            _ = deduplicate_chats(chats, deduplicated)

        # Matching checkpoints are resumed after a failed moderation run.
        _ = moderate_dataset(deduplicated, koala_predictions, text_field="content", batch_size=128)
    _ = guided_sample_dataset(
        koala_predictions, sample,
        {"OK": 49_911, "H": 20_000, "V": 10_000, "S": 8_000,
         "SH": 7_000, "HR": 5_088, "H2": 1},
        seed=42,
    )
    print("Pipeline complete.", flush=True)


def main() -> None:
    pipeline()


if __name__ == "__main__":
    main()
