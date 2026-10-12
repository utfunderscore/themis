"""Flushed stage logging for local and remote decision runs."""

from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from threading import Event, Thread
from time import perf_counter


def log(message: str) -> None:
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    print(f"[{timestamp}] {message}", flush=True)


@contextmanager
def stage(name: str) -> Iterator[None]:
    """Report long blocking operations without suggesting measurable progress."""
    started = perf_counter()
    done = Event()

    def heartbeat() -> None:
        while not done.wait(15):
            log(f"{name}: still running; elapsed={perf_counter() - started:.1f}s")

    log(f"{name}: starting")
    thread = Thread(target=heartbeat, daemon=True)
    thread.start()
    try:
        yield
    except BaseException as exc:
        done.set()
        thread.join()
        log(f"{name}: interrupted/failed ({type(exc).__name__}); "
            f"elapsed={perf_counter() - started:.1f}s")
        raise
    else:
        done.set()
        thread.join()
        log(f"{name}: complete; elapsed={perf_counter() - started:.1f}s")


def log_count(name: str, completed: int, total: int, started: float) -> None:
    elapsed = perf_counter() - started
    rate = completed / elapsed if elapsed > 0 else 0
    eta = (total - completed) / rate if rate > 0 else 0
    log(f"{name}: {completed}/{total} ({completed / total:.0%}); "
        f"elapsed={elapsed:.1f}s; rate={rate:.2f} rows/s; ETA={eta:.1f}s")
