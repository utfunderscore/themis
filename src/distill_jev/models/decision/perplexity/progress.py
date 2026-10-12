"""Flushed stage logging for local and remote decision runs."""

from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
import os
from threading import Event, Thread
from time import perf_counter, time

RUN_STARTED_AT_ENV = "DISTILL_JEV_RUN_STARTED_AT"


def log(message: str, *, started_at: float | None = None) -> None:
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    if started_at is None and RUN_STARTED_AT_ENV in os.environ:
        started_at = float(os.environ[RUN_STARTED_AT_ENV])
    elapsed = f" [run elapsed={max(0, time() - started_at):.1f}s]" if started_at is not None else ""
    print(f"[{timestamp}]{elapsed} {message}", flush=True)


@contextmanager
def stage(name: str) -> Iterator[None]:
    """Report long blocking operations without suggesting measurable progress."""
    started = perf_counter()
    done = Event()

    def heartbeat() -> None:
        while not done.wait(15):
            log(f"{name}: still running")

    log(f"{name}: starting")
    thread = Thread(target=heartbeat, daemon=True)
    thread.start()
    try:
        yield
    except BaseException as exc:
        done.set()
        thread.join()
        log(f"{name}: interrupted/failed ({type(exc).__name__}); "
            f"stage duration={perf_counter() - started:.1f}s")
        raise
    else:
        done.set()
        thread.join()
        log(f"{name}: complete; stage duration={perf_counter() - started:.1f}s")
