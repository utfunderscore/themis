"""Batch moderation REST API; install the ``api`` and ``training`` extras."""

from collections.abc import Mapping, Sequence
from contextlib import asynccontextmanager
import os
from pathlib import Path
from threading import Lock
from typing import Annotated, Protocol

from fastapi import Body, FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field, StrictStr

PAGE = Path(__file__).parent / "static" / "index.html"


class Predictor(Protocol):
    def predict(
        self, texts: list[str], batch_size: int = 32,
    ) -> Sequence[Mapping[str, object]]: ...


Probability = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
Chat = Annotated[StrictStr, Field(min_length=1, max_length=16384)]


class ChatScores(BaseModel):
    category: str
    action: str
    category_probabilities: dict[str, Probability]
    action_probabilities: dict[str, Probability]


def create_app(
    model_directory: str | Path | None = None,
    *,
    device: str | None = None,
    max_length: int | None = None,
    batch_size: int = 32,
    predictor: Predictor | None = None,
) -> FastAPI:
    """Load one model per app at startup, or use an injected predictor.

    Without a model directory argument, read ``THEMIS_MODEL_DIRECTORY``.
    ``THEMIS_DEVICE`` optionally selects the device. Concurrent requests share
    the model serially; inference runs in FastAPI's worker thread pool.
    """
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    lock = Lock()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if predictor is None:
            directory = model_directory or os.environ.get("THEMIS_MODEL_DIRECTORY")
            if not directory:
                raise ValueError("Set THEMIS_MODEL_DIRECTORY or pass model_directory")
            from .models.deberta.inference import Moderation

            app.state.predictor = Moderation(
                directory, device=device or os.environ.get("THEMIS_DEVICE"),
                max_length=max_length,
            )
        else:
            app.state.predictor = predictor
        try:
            yield
        finally:
            del app.state.predictor

    app = FastAPI(
        title="Themis Batch Inference API", version="1.0.0", lifespan=lifespan,
    )

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def tester() -> HTMLResponse:
        """Serve the browser test page for trying the classifier."""
        return HTMLResponse(PAGE.read_text(encoding="utf-8"))

    @app.post("/v1/inference", response_model=list[ChatScores], operation_id="batchInference")
    def infer(chats: Annotated[list[Chat], Body(max_length=256)]) -> list[ChatScores]:
        """Score up to 256 chats, returning one result per chat in input order.

        Category and action probabilities are separate distributions. Each
        chat is truncated to the model's configured token limit. An empty
        input array returns an empty array.
        """
        if not chats:
            return []
        with lock:
            predictions = app.state.predictor.predict(chats, batch_size=batch_size)
        if len(predictions) != len(chats):
            raise RuntimeError("Predictor returned an incorrect number of results")
        return [ChatScores.model_validate(result) for result in predictions]

    return app
