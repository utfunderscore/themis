"""Publish a trained student, and optionally its ONNX bundle, to the Hugging Face Hub.

Requires the ``training`` extra (``huggingface_hub`` ships with Transformers).
Authenticate first with ``hf auth login`` or pass ``token``.
"""

import json
from pathlib import Path

MODEL_FILES = (
    "model.safetensors", "config.json", "tokenizer_config.json",
    "special_tokens_map.json", "added_tokens.json", "spm.model",
)
# Self-contained bundle, as read by the Java library's ModelBundle.
ONNX_FILES = (
    "model.onnx", "inference_config.json", "config.json", "tokenizer_config.json",
    "special_tokens_map.json", "added_tokens.json", "spm.model",
)
MODEL_CARD = Path(__file__).with_name("model_card.md")


def upload_model(
    model_directory: str | Path,
    repo_id: str,
    *,
    onnx_directory: str | Path | None = None,
    max_length: int | None = None,
    private: bool = True,
    token: str | None = None,
    commit_message: str = "Upload Themis DeBERTa moderation model",
) -> str:
    """Create ``repo_id`` if needed, upload the model and card; return the commit URL.

    ``model_directory`` accepts a training output directory or its ``model/``
    subdirectory. ONNX files go under ``onnx/`` when ``onnx_directory`` is given.
    Training-only files such as ``training_args.bin`` are not uploaded.
    ``max_length`` defaults to the adjacent ``run_config.json``; it is written to
    the uploaded tokenizer configs so Hub tokenizers truncate like training.
    """
    from huggingface_hub import CommitOperationAdd, HfApi

    directory = Path(model_directory)
    if (directory / "model").is_dir():
        directory = directory / "model"
    sources = {name: directory / name for name in MODEL_FILES}
    if onnx_directory is not None:
        sources |= {f"onnx/{name}": Path(onnx_directory) / name for name in ONNX_FILES}
    missing = [str(path) for path in sources.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing model files: {', '.join(missing)}")
    if max_length is None:
        config_path = directory.parent / "run_config.json"
        if not config_path.is_file():
            raise ValueError("Supply max_length when run_config.json is unavailable")
        max_length = json.loads(config_path.read_text(encoding="utf-8"))["max_length"]
    if onnx_directory is not None:
        exported = json.loads(sources["onnx/inference_config.json"].read_text(encoding="utf-8"))["max_length"]
        if exported != max_length:
            raise ValueError(f"ONNX bundle max_length {exported} does not match {max_length}")

    api = HfApi(token=token)
    api.create_repo(repo_id, repo_type="model", private=private, exist_ok=True)
    card = MODEL_CARD.read_text(encoding="utf-8").replace("{{repo_id}}", repo_id)
    operations = []
    for name, source in sources.items():
        content: Path | bytes = source
        if source.name == "tokenizer_config.json":
            tokenizer_config = json.loads(source.read_text(encoding="utf-8"))
            tokenizer_config["model_max_length"] = max_length
            content = (json.dumps(tokenizer_config, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
        operations.append(CommitOperationAdd(path_in_repo=name, path_or_fileobj=content))
    operations.append(CommitOperationAdd("README.md", card.encode("utf-8")))
    commit = api.create_commit(
        repo_id, operations, commit_message=commit_message, repo_type="model",
    )
    return commit.commit_url
