"""Timestamped, immediately flushed console progress for the card pipeline."""

from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from time import perf_counter


def report(component: str, message: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] [{component}] {message}", flush=True)


@contextmanager
def stage_progress(stage: str, position: int, total: int) -> Iterator[None]:
    started = perf_counter()
    report(stage, f"AVVIO stadio {position}/{total}")
    try:
        yield
    except KeyboardInterrupt:
        report(stage, f"INTERROTTO dopo {perf_counter() - started:.1f}s")
        raise
    except Exception as error:
        report(stage, f"ERRORE dopo {perf_counter() - started:.1f}s: {error}")
        raise
    else:
        report(stage, f"COMPLETATO in {perf_counter() - started:.1f}s")
