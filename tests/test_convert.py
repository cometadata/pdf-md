from pdf_md.config import ExtractConfig
from pdf_md.pdf_input import DocInput
from pdf_md.types import PageResult
from pdf_md.convert import _extract_isolated


def test_extract_isolated_success(sample_pdf):
    doc = DocInput(doc_id="sample", source=str(sample_pdf), path=str(sample_pdf))
    result = _extract_isolated(doc, ExtractConfig(use_ocr=False), timeout=120)
    assert result.error is None
    assert len(result.pages) >= 1


def test_extract_isolated_corrupt_returns_error(tmp_path):
    bad = tmp_path / "bad.pdf"
    bad.write_bytes(b"not a pdf")
    doc = DocInput(doc_id="bad", source=str(bad), path=str(bad))
    result = _extract_isolated(doc, ExtractConfig(use_ocr=False), timeout=120)
    assert result.error is not None


from pdf_md.convert import _WorkerPool


def test_worker_pool_processes_all_docs(sample_pdf_dir):
    from pdf_md.pdf_input import load_docs
    docs = list(load_docs(str(sample_pdf_dir)))
    cfg = ExtractConfig(workers=2, use_ocr=False)
    pool = _WorkerPool(cfg)
    pool.start()
    for d in docs:
        pool.submit(d)
    pool.finish()  # enqueue sentinels
    pool.mark_input_done()
    results = []
    while True:
        r = pool.get(timeout=120)
        if r is None:
            break
        results.append(r)
    pool.join()
    assert sorted(r.doc_id for r in results) == ["doc_a", "doc_b"]
    assert all(r.error is None for r in results)


from pdf_md.convert import convert_docs, convert_docs_streaming, _group_by_document


def test_convert_docs_streaming_yields_rows(sample_pdf_dir):
    from pdf_md.pdf_input import load_docs
    docs = load_docs(str(sample_pdf_dir))
    cfg = ExtractConfig(workers=2, worker_batch_size=1, use_ocr=False)
    batches = list(convert_docs_streaming(docs, cfg))
    all_rows = [row for batch in batches for row in batch]
    doc_ids = {row[0] for row in all_rows}
    assert doc_ids == {"doc_a", "doc_b"}
    # every row is (doc_id, source, PageResult, error)
    for doc_id, source, page_result, error in all_rows:
        assert isinstance(page_result, PageResult)


def test_convert_docs_groups_by_document(sample_pdf_dir):
    from pdf_md.pdf_input import load_docs
    docs = load_docs(str(sample_pdf_dir))
    results = convert_docs(docs, ExtractConfig(workers=2, use_ocr=False))
    assert sorted(r.doc_id for r in results) == ["doc_a", "doc_b"]
    for r in results:
        idxs = [p.page_index for p in r.pages]
        assert idxs == sorted(idxs)


def test_convert_docs_max_docs_cap(sample_pdf_dir):
    from pdf_md.pdf_input import load_docs
    docs = load_docs(str(sample_pdf_dir))
    results = convert_docs(docs, ExtractConfig(workers=2, use_ocr=False), max_docs=1)
    assert len(results) == 1


import queue as _queue
import time as _time


class _FakeProc:
    """Stand-in for a pool worker process: starts alive, dies when killed."""

    exitcode = None

    def __init__(self):
        self.killed = False

    def is_alive(self):
        return not self.killed

    def kill(self):
        self.killed = True

    def join(self, timeout=None):
        pass


def test_scan_once_recovers_hung_worker():
    """A worker stuck on one doc past doc_timeout is killed and its doc gets a
    timeout-error result — so the consumer loop can't spin forever."""
    cfg = ExtractConfig(workers=1, doc_timeout=5, use_ocr=False)
    pool = _WorkerPool(cfg)
    try:
        fp = _FakeProc()
        pool._procs[0] = fp
        # in-flight doc started 999s ago -> well past doc_timeout=5 -> hung
        pool._current[0] = ("docX", _time.time() - 999)
        pool._docs_by_id["docX"] = DocInput(
            doc_id="docX", source="s.pdf", path="/nonexistent",
        )
        pool._submitted = 1
        spawned = []
        pool._spawn = lambda idx: spawned.append(idx)

        pool._scan_once()

        assert fp.killed is True
        assert spawned == [0]
        result = pool.result_q.get(timeout=2)
        assert result.doc_id == "docX"
        assert result.error and "timeout" in result.error.lower()
        assert result.pages == []
    finally:
        pool._manager.shutdown()


def test_scan_once_leaves_healthy_worker_alone():
    """A worker that recently picked up a doc must not be killed or respawned."""
    cfg = ExtractConfig(workers=1, doc_timeout=600, use_ocr=False)
    pool = _WorkerPool(cfg)
    try:
        pool._procs[0] = _FakeProc()
        pool._current[0] = ("docX", _time.time())  # just started
        pool._docs_by_id["docX"] = DocInput(doc_id="docX", source="s", path="/x")
        pool._submitted = 1

        def _no_spawn(idx):
            raise AssertionError("healthy worker should not be respawned")

        pool._spawn = _no_spawn
        pool._scan_once()  # must not raise

        try:
            pool.result_q.get(timeout=0.3)
            raise AssertionError("no result should be emitted for a healthy worker")
        except _queue.Empty:
            pass
    finally:
        pool._manager.shutdown()
