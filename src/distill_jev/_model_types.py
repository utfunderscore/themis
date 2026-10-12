"""The small inference API used from libraries installed only in Modal images.

These structural interfaces describe our pinned model's two-dimensional scores,
not the full Torch/Transformers APIs. Casts at module import boundaries keep
remote-only dependencies out of the local environment and the rest fully typed.
"""

from collections.abc import Iterable, Mapping
from contextlib import AbstractContextManager
from typing import Protocol, Self


class ScoreTensor(Protocol):
    def softmax(self, dim: int) -> Self: ...
    def cpu(self) -> Self: ...
    def tolist(self) -> list[list[float]]: ...
    def float(self) -> Self: ...


class TokenBatch(Protocol):
    def to(self, device: str) -> Self: ...
    def keys(self) -> Iterable[str]: ...
    def __getitem__(self, key: str) -> object: ...


class Tokenizer(Protocol):
    def __call__(
        self, texts: list[str], *, truncation: bool, max_length: int,
    ) -> Mapping[str, list[list[int]]]: ...
    def pad(
        self, encoded_inputs: list[dict[str, list[int]]], *, padding: bool,
        return_tensors: str,
    ) -> TokenBatch: ...


class ModelConfig(Protocol):
    @property
    def id2label(self) -> Mapping[int, str]: ...


class ModelOutput(Protocol):
    @property
    def logits(self) -> ScoreTensor: ...


class Model(Protocol):
    def half(self) -> Self: ...
    @property
    def config(self) -> ModelConfig: ...
    def to(self, device: str) -> Self: ...
    def eval(self) -> Self: ...
    def __call__(self, **inputs: object) -> ModelOutput: ...


class TokenizerFactory(Protocol):
    def from_pretrained(self, path: str, *, local_files_only: bool) -> Tokenizer: ...


class ModelFactory(Protocol):
    def from_pretrained(
        self, path: str, *, local_files_only: bool, use_safetensors: bool,
    ) -> Model: ...


class TransformersModule(Protocol):
    @property
    def AutoTokenizer(self) -> TokenizerFactory: ...
    @property
    def AutoModelForSequenceClassification(self) -> ModelFactory: ...


class TorchModule(Protocol):
    def inference_mode(self) -> AbstractContextManager[object]: ...


class HubModule(Protocol):
    def snapshot_download(
        self, repo_id: str, *, revision: str, local_dir: str, allow_patterns: list[str],
    ) -> str: ...
