from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from typing import Iterator


def _format_elapsed(seconds: float) -> str:
    total_seconds = max(float(seconds), 0.0)
    minutes = int(total_seconds // 60)
    secs = total_seconds - minutes * 60
    return f"{minutes}m {secs:.1f}s"


@contextmanager
def heartbeat(label: str, interval_s: float = 30.0) -> Iterator[None]:
    start = time.perf_counter()
    stop_event = threading.Event()

    def _heartbeat_worker() -> None:
        while not stop_event.wait(interval_s):
            elapsed = time.perf_counter() - start
            print(f"{label} still running... elapsed={_format_elapsed(elapsed)}", flush=True)

    print(f"{label} started", flush=True)
    worker = threading.Thread(target=_heartbeat_worker, daemon=True)
    worker.start()

    try:
        yield
    except Exception:
        elapsed = time.perf_counter() - start
        print(f"{label} failed after {_format_elapsed(elapsed)}", flush=True)
        raise
    else:
        elapsed = time.perf_counter() - start
        print(f"{label} finished in {_format_elapsed(elapsed)}", flush=True)
    finally:
        stop_event.set()
        worker.join(timeout=0.1)
