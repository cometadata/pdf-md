from __future__ import annotations

import logging
import multiprocessing as mp
import os
import queue
import threading
import time
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

# This package ships a module named ``types.py`` at its root. A spawned
# subprocess is launched as a plain ``python -c`` that inherits the parent's
# cwd; if that cwd is the package directory, ``''`` lands on ``sys.path`` and
# ``pdf_md/types.py`` shadows the stdlib ``types`` module, breaking the child's
# own ``import multiprocessing``. ``PYTHONSAFEPATH`` keeps cwd off the child's
# path. ``setdefault`` so an explicit user setting still wins.
os.environ.setdefault("PYTHONSAFEPATH", "1")

from . import diagnostics
from .config import ExtractConfig
from .pdf_input import DocInput
from .types import ConversionResult, PageResult
from .worker import extract_doc

LOGGER = logging.getLogger(__name__)

_MP = mp.get_context("spawn")

# A flat output row: (doc_id, source, PageResult, error_or_None)
Row = Tuple[str, str, PageResult, Optional[str]]


def _isolated_target(doc: DocInput, config: ExtractConfig, out: "mp.Queue") -> None:
    out.put(extract_doc(doc, config))


def _extract_isolated(
    doc: DocInput, config: ExtractConfig, timeout: int,
) -> ConversionResult:
    out: "mp.Queue" = _MP.Queue()
    proc = _MP.Process(target=_isolated_target, args=(doc, config, out))
    proc.start()
    proc.join(timeout)

    if proc.is_alive():
        LOGGER.warning("Doc %s timed out after %ds; killing", doc.doc_id, timeout)
        proc.kill()
        proc.join(5)
        return ConversionResult(
            doc_id=doc.doc_id, source=doc.source, pages=[],
            error=f"timeout after {timeout}s",
        )
    try:
        return out.get_nowait()
    except queue.Empty:
        return ConversionResult(
            doc_id=doc.doc_id, source=doc.source, pages=[],
            error=f"worker crashed (exitcode={proc.exitcode})",
        )


def _pool_worker(
    idx: int,
    config: ExtractConfig,
    task_q: "mp.Queue",
    result_q: "mp.Queue",
    current: dict,
) -> None:
    import faulthandler
    faulthandler.enable()
    while True:
        with diagnostics.timed("worker.idle_waiting_for_task"):
            doc = task_q.get()
        if doc is None:
            return
        # (doc_id, start_time): lets the watcher spot a worker hung on one doc.
        current[idx] = (doc.doc_id, time.time())
        try:
            with diagnostics.timed("worker.busy_extract"):
                result = extract_doc(doc, config)
            result_q.put(result)
        finally:
            current[idx] = None


class _WorkerPool:
    def __init__(self, config: ExtractConfig) -> None:
        self.config = config
        self.n = config.effective_workers()
        self.task_q: "mp.Queue" = _MP.Queue(maxsize=self.n * 4)
        self.result_q: "mp.Queue" = _MP.Queue()
        self._manager = _MP.Manager()
        self._current = self._manager.dict()       # idx -> doc_id | None
        self._docs_by_id: dict = {}                # doc_id -> DocInput (in flight)
        self._procs: dict = {}                     # idx -> Process
        self._submitted = 0
        self._returned = 0
        self._lock = threading.Lock()
        self._done = threading.Event()
        self._watcher = threading.Thread(target=self._watch, daemon=True)

    def _spawn(self, idx: int) -> None:
        p = _MP.Process(
            target=_pool_worker,
            args=(idx, self.config, self.task_q, self.result_q, self._current),
            daemon=True,
        )
        p.start()
        self._procs[idx] = p

    def start(self) -> None:
        for idx in range(self.n):
            self._current[idx] = None
            self._spawn(idx)
        self._watcher.start()

    def submit(self, doc: DocInput) -> None:
        with self._lock:
            self._docs_by_id[doc.doc_id] = doc
            self._submitted += 1
        self.task_q.put(doc)

    def finish(self) -> None:
        for _ in range(self.n):
            self.task_q.put(None)

    def get(self, timeout: float) -> Optional[ConversionResult]:
        while True:
            with self._lock:
                if self._returned >= self._submitted and self._submitted_complete():
                    return None
            try:
                result = self.result_q.get(timeout=timeout)
            except queue.Empty:
                continue
            with self._lock:
                self._returned += 1
                self._docs_by_id.pop(result.doc_id, None)
            return result

    def _submitted_complete(self) -> bool:
        # set True by streaming generator once finish() has been called
        return self._done.is_set()

    def mark_input_done(self) -> None:
        self._done.set()

    # Cadence of the in-flight progress line. Silent during healthy operation
    # (most arXiv docs convert in <1s), loud once a worker is meaningfully
    # behind so an operator can distinguish a self-healing job from a wedged
    # one.
    _LOG_EVERY = 30.0
    _LOG_FLOOR = 30.0

    def _watch(self) -> None:
        last_log = time.time()
        while not self._done.is_set() or self._returned < self._submitted:
            time.sleep(0.5)
            self._scan_once()
            now = time.time()
            if now - last_log >= self._LOG_EVERY:
                last_log = now
                self._log_inflight(now)

    def _log_inflight(self, now: float) -> None:
        worst_idx, worst_doc, worst_age = -1, None, 0.0
        for idx in list(self._procs):
            entry = self._current.get(idx)
            if entry is None:
                continue
            age = now - entry[1]
            if age > worst_age:
                worst_idx, worst_doc, worst_age = idx, entry[0], age
        if worst_doc is None or worst_age < self._LOG_FLOOR:
            return
        LOGGER.info(
            "watchdog: worker %d on doc=%s for %.0fs (kills at %ds); "
            "pool=%d/%d returned",
            worst_idx, worst_doc, worst_age, self.config.doc_timeout,
            self._returned, self._submitted,
        )

    def _scan_once(self) -> None:
        timeout = self.config.doc_timeout
        now = time.time()
        for idx, proc in list(self._procs.items()):
            entry = self._current.get(idx)
            alive = proc.is_alive()
            hung = alive and entry is not None and now - entry[1] > timeout
            if alive and not hung:
                continue
            if not alive and proc.exitcode == 0:
                continue  # clean sentinel exit

            lost_doc_id = entry[0] if entry else None
            if hung:
                LOGGER.warning(
                    "Worker %d hung on doc=%s (>%ds); killing",
                    idx, lost_doc_id, timeout,
                )
                proc.kill()
                proc.join(5)
            else:
                LOGGER.warning(
                    "Worker %d died (exitcode=%s), in-flight doc=%s",
                    idx, proc.exitcode, lost_doc_id,
                )
            # Clear before respawn so the replacement starts with a clean slot.
            self._current[idx] = None
            self._spawn(idx)

            if not lost_doc_id:
                continue
            with self._lock:
                doc = self._docs_by_id.get(lost_doc_id)
            if doc is None:
                continue
            if hung:
                result = ConversionResult(
                    doc_id=doc.doc_id, source=doc.source, pages=[],
                    error=f"timeout: hung pool worker >{timeout}s",
                )
            else:
                result = _extract_isolated(doc, self.config, timeout)
                if result.error is not None:
                    result.error = f"crashed in pool; {result.error}"
            self.result_q.put(result)

    def join(self) -> None:
        for p in self._procs.values():
            p.join(timeout=10)
        # Stop the watcher before tearing down the Manager: _scan_once reads
        # the Manager-backed _current dict every pass, and touching it after
        # shutdown raises BrokenPipeError in the watcher thread. By now the
        # watcher's loop condition is already false, so this returns promptly.
        if self._watcher.ident is not None:
            self._watcher.join(timeout=5)
        self._manager.shutdown()


from collections import OrderedDict


def _rows_from_result(result: ConversionResult) -> List[Row]:
    if result.error is not None:
        return [(
            result.doc_id, result.source,
            PageResult(page_index=-1, content=""), result.error,
        )]
    return [
        (result.doc_id, result.source, page, None)
        for page in result.pages
    ]


def _group_by_document(rows: List[Row]) -> List[ConversionResult]:
    docs: "OrderedDict[str, ConversionResult]" = OrderedDict()
    for doc_id, source, page_result, error in rows:
        if doc_id not in docs:
            docs[doc_id] = ConversionResult(doc_id=doc_id, source=source, pages=[])
        if error is not None:
            docs[doc_id].error = error
        else:
            docs[doc_id].pages.append(page_result)
    for d in docs.values():
        d.pages.sort(key=lambda p: p.page_index)
    return list(docs.values())


def convert_docs_streaming(
    docs: Iterator[DocInput],
    config: ExtractConfig,
    *,
    max_docs: Optional[int] = None,
) -> Iterator[List[Row]]:
    pool = _WorkerPool(config)
    pool.start()

    start = time.time()
    submitted = 0

    def _feed() -> None:
        nonlocal submitted
        for doc in docs:
            if max_docs is not None and submitted >= max_docs:
                break
            pool.submit(doc)
            submitted += 1
        pool.finish()
        pool.mark_input_done()

    feeder = threading.Thread(target=_feed, daemon=True)
    feeder.start()

    batch: List[Row] = []
    docs_in_batch = 0
    processed = 0
    try:
        while True:
            result = pool.get(timeout=config.doc_timeout + 60)
            if result is None:
                break
            if processed == 0 and diagnostics.enabled():
                LOGGER.info(
                    "[PDF_MD_DIAG] time-to-first-result: %.2fs", time.time() - start
                )
            batch.extend(_rows_from_result(result))
            docs_in_batch += 1
            processed += 1
            if docs_in_batch >= config.worker_batch_size:
                yield batch
                batch, docs_in_batch = [], 0
        if batch:
            yield batch
    finally:
        feeder.join(timeout=10)
        pool.join()

    elapsed = time.time() - start
    LOGGER.info(
        "Converted %d docs in %.1fs (%.2f docs/s)",
        processed, elapsed, processed / elapsed if elapsed else 0,
    )


def convert_docs(
    docs: Iterator[DocInput],
    config: ExtractConfig,
    *,
    max_docs: Optional[int] = None,
) -> List[ConversionResult]:
    all_rows: List[Row] = []
    for batch in convert_docs_streaming(docs, config, max_docs=max_docs):
        all_rows.extend(batch)
    return _group_by_document(all_rows)
