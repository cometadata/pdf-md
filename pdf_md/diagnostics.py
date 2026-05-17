from __future__ import annotations

import atexit
import os
import sys
import threading
import time
from contextlib import contextmanager
from typing import Callable, Dict, Iterator, Tuple

_ENV_VAR = "PDF_MD_DIAG"


def enabled() -> bool:
    return os.environ.get(_ENV_VAR, "").strip().lower() in {"1", "true", "yes"}


class Accumulator:

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._data: Dict[str, list] = {}  # name -> [count, total_s]

    def record(self, name: str, seconds: float) -> None:
        with self._lock:
            slot = self._data.setdefault(name, [0, 0.0])
            slot[0] += 1
            slot[1] += seconds

    def snapshot(self) -> Dict[str, Tuple[int, float]]:
        with self._lock:
            return {k: (int(c), float(t)) for k, (c, t) in self._data.items()}

    def report(self, title: str) -> str:
        snap = self.snapshot()
        if not snap:
            return f"{title}: (no samples)"
        width = max(len(k) for k in snap)
        lines = [f"{title}:"]
        for name, (count, total) in sorted(snap.items(), key=lambda kv: -kv[1][1]):
            avg_ms = (total / count * 1e3) if count else 0.0
            lines.append(
                f"  {name:<{width}}  n={count:>7}  "
                f"total={total:>9.3f}s  avg={avg_ms:>8.3f}ms"
            )
        return "\n".join(lines)


# One accumulator per process. Spawned workers each get their own.
_ACC = Accumulator()


def accumulator() -> Accumulator:
    return _ACC


@contextmanager
def timed(name: str) -> Iterator[None]:
    if not enabled():
        yield
        return
    t0 = time.perf_counter()
    try:
        yield
    finally:
        _ACC.record(name, time.perf_counter() - t0)


def _wrap_timed(name: str, fn: Callable) -> Callable:

    def _wrapped(*args, **kwargs):
        t0 = time.perf_counter()
        try:
            return fn(*args, **kwargs)
        finally:
            _ACC.record(name, time.perf_counter() - t0)

    _wrapped._pdf_md_diag_wrapped = True  # type: ignore[attr-defined]
    _wrapped._pdf_md_diag_orig = fn  # type: ignore[attr-defined]
    return _wrapped


def dump() -> None:
    # Goes straight to stderr, not the logging module: spawned worker
    # processes don't inherit the main process's logging config, so an
    # INFO-level log here would silently vanish. HF Jobs captures stderr.
    report = _ACC.report(f"[PDF_MD_DIAG pid={os.getpid()}]")
    print(report, file=sys.stderr, flush=True)


_installed = False


def install() -> None:
    # Order matters: the onnxruntime constructor patch must be in place before
    # the layout module is imported, because importing pymupdf.layout runs
    # activate() which builds the ONNX sessions we want to instrument.
    global _installed
    if not enabled() or _installed:
        return
    _installed = True
    _instrument_onnxruntime()
    _instrument_layout()
    atexit.register(dump)
    print(
        f"[PDF_MD_DIAG] instrumentation installed (pid={os.getpid()})",
        file=sys.stderr, flush=True,
    )


def _instrument_onnxruntime() -> None:
    try:
        import onnxruntime as ort
    except ImportError:  # pragma: no cover - onnxruntime always present in prod
        return
    ctor = ort.InferenceSession
    if getattr(ctor, "_pdf_md_diag", False):
        return

    def _diag_ctor(*args, **kwargs):
        session = ctor(*args, **kwargs)
        model = args[0] if args else kwargs.get("path_or_bytes")
        label = os.path.basename(model) if isinstance(model, str) else "onnx_session"
        run = getattr(session, "run", None)
        if callable(run) and not getattr(run, "_pdf_md_diag_wrapped", False):
            try:
                session.run = _wrap_timed(f"onnx.run[{label}]", run)
            except (AttributeError, TypeError):  # pragma: no cover - C-level session
                pass
        return session

    _diag_ctor._pdf_md_diag = True  # type: ignore[attr-defined]
    ort.InferenceSession = _diag_ctor


def _instrument_layout() -> None:
    # Importing this module triggers pymupdf.layout.activate(); that is fine
    # and intended — the ONNX sessions it builds get caught by the constructor
    # patch installed just above.
    try:
        from pymupdf.layout.onnx import BoxRFDGNN as mod
    except ImportError:  # pragma: no cover - layout always present in prod
        print(
            "[PDF_MD_DIAG] pymupdf.layout not importable; layout uninstrumented",
            file=sys.stderr, flush=True,
        )
        return

    targets = {
        "create_input_data_from_page": "layout.input_data_from_page",
        "get_nn_input_from_datadict": "layout.graph_construction",
    }
    for attr, label in targets.items():
        fn = getattr(mod, attr, None)
        if callable(fn) and not getattr(fn, "_pdf_md_diag_wrapped", False):
            setattr(mod, attr, _wrap_timed(label, fn))

    predict = getattr(mod.BoxRFDGNN, "predict", None)
    if callable(predict) and not getattr(predict, "_pdf_md_diag_wrapped", False):
        mod.BoxRFDGNN.predict = _wrap_timed("layout.predict_total", predict)
