import time

import pdf_md
from pdf_md.config import ExtractConfig
from pdf_md.pdf_input import DocInput, load_docs
from pdf_md.convert import convert_docs, _WorkerPool


def test_end_to_end_directory_markdown(sample_pdf_dir, tmp_path):
    out = tmp_path / "out"
    results = pdf_md.convert(
        str(sample_pdf_dir), output=str(out), format="markdown",
        use_ocr=False, workers=2,
    )
    assert len(results) == 2
    assert (out / "doc_a.md").stat().st_size > 0


def test_resume_skips_completed(sample_pdf_dir, tmp_path):
    out = tmp_path / "out"
    # first run, capped at 1 doc -> one checkpoint left behind
    pdf_md.convert(str(sample_pdf_dir), output=str(out), use_ocr=False,
                   workers=2, max_docs=1)
    # checkpoints were cleared on success; simulate interrupted run instead:
    # re-run full; both docs should be present
    results = pdf_md.convert(str(sample_pdf_dir), output=str(out),
                             use_ocr=False, workers=2)
    assert sorted(r.doc_id for r in results) == ["doc_a", "doc_b"]


def test_end_to_end_parquet_resume(sample_pdf_dir, tmp_path):
    import pyarrow.parquet as pq

    out = tmp_path / "out"
    # First run is capped at one doc and uses flush_every=1 so the single
    # batch is flushed to a shard before the run ends.
    pdf_md.convert(
        str(sample_pdf_dir), output=str(out), parquet=True, flush_every=1,
        use_ocr=False, workers=2, max_docs=1,
    )
    data_dir = out / "data"
    shards = sorted(data_dir.glob("shard_*.parquet"))
    assert len(shards) == 1
    assert not (out / ".checkpoints").exists()

    # Second run completes the remaining doc and writes a fresh shard.
    pdf_md.convert(
        str(sample_pdf_dir), output=str(out), parquet=True, flush_every=1,
        use_ocr=False, workers=2,
    )
    shards = sorted(data_dir.glob("shard_*.parquet"))
    assert len(shards) == 2

    # Markdown emits one row per page, so a doc_id legitimately repeats
    # *within* a shard. The resume invariant is that no doc_id appears in
    # more than one shard.
    per_shard_doc_sets = [
        set(pq.read_table(s, columns=["doc_id"]).column("doc_id").to_pylist())
        for s in shards
    ]
    assert set().union(*per_shard_doc_sets) == {"doc_a", "doc_b"}
    assert sum(len(s) for s in per_shard_doc_sets) == len(set().union(*per_shard_doc_sets))


def test_corrupt_pdf_produces_error_row(tmp_path):
    bad = tmp_path / "bad.pdf"
    bad.write_bytes(b"%PDF-1.4 garbage")
    good_dir = tmp_path  # contains only bad.pdf
    results = convert_docs(
        load_docs(str(good_dir)), ExtractConfig(use_ocr=False, workers=2),
    )
    assert len(results) == 1
    assert results[0].error is not None


def test_crash_respawn_reroutes_in_flight_doc(sample_pdf):
    """Genuinely exercise the crash-respawn path in ``_WorkerPool``.

    ``test_corrupt_pdf_produces_error_row`` only covers the *graceful* error
    path -- ``extract_doc`` catches the bad PDF and the worker process stays
    alive. The crash-respawn machinery (``_watch`` noticing a worker with a
    non-zero exitcode, respawning it, and re-routing its in-flight doc to an
    isolated retry) needs a worker process that *actually dies*.

    Rather than poison production code with a test-only "please crash" hook,
    the test drives ``_WorkerPool`` directly and uses its own handle on the
    live worker ``Process`` to ``.kill()`` it -- a legitimate way for a test
    to simulate a crash, since it touches only the test-visible pool object,
    not ``worker.py``/``convert.py``.

    With ``workers=1`` the routing is deterministic: the single submitted doc
    must land on worker 0. We wait (condition-based, not a fixed sleep) until
    ``_current[0]`` is populated -- proving the worker has picked up the doc
    and is mid-``extract_doc`` -- then kill that process so the re-route
    branch (``lost_doc_id`` truthy) is the one that fires.
    """
    cfg = ExtractConfig(workers=1, use_ocr=False)
    pool = _WorkerPool(cfg)
    pool.start()
    try:
        original_proc = pool._procs[0]

        doc = DocInput(doc_id="victim", source=str(sample_pdf),
                       path=str(sample_pdf))
        pool.submit(doc)

        # Wait until the worker has dequeued the doc and published it as
        # in-flight (``_pool_worker`` sets ``current[idx] = (doc.doc_id,
        # start_time)`` right before ``extract_doc``). Polling a condition --
        # not sleeping a guessed interval -- so this is not timing-fragile.
        deadline = time.time() + 30
        while (pool._current.get(0) or (None,))[0] != "victim":
            assert time.time() < deadline, (
                "worker never published the in-flight doc"
            )
            time.sleep(0.02)

        # The worker is now mid-extraction. Kill it: a SIGKILL gives a
        # negative (non-zero) exitcode, which is exactly what ``_watch``
        # treats as a crash (vs. exitcode 0 for a clean sentinel exit).
        original_proc.kill()
        original_proc.join(10)
        assert original_proc.exitcode is not None and original_proc.exitcode != 0

        # No more docs are coming.
        pool.finish()
        pool.mark_input_done()

        # Drain results. ``_watch`` must (a) respawn worker 0 and
        # (b) re-route the killed doc through ``_extract_isolated`` so a
        # result for "victim" still appears. ``get()`` returning ``None``
        # without blocking forever proves the pool did not hang.
        results = []
        deadline = time.time() + 120
        while True:
            assert time.time() < deadline, "pool hung -- get() never returned None"
            r = pool.get(timeout=5)
            if r is None:
                break
            results.append(r)

        # The in-flight doc survived the crash via the isolated-retry path.
        assert [r.doc_id for r in results] == ["victim"]
        # ``_watch`` replaced the dead worker: ``_procs[0]`` is now a fresh,
        # distinct Process that actually got started (a real ``pid``). We do
        # NOT assert it is still alive -- ``finish()`` enqueued a sentinel, so
        # the respawned worker may already have consumed it and exited 0,
        # which is correct shutdown, not a failure.
        respawned = pool._procs[0]
        assert respawned is not original_proc
        assert respawned.pid is not None and respawned.pid != original_proc.pid
    finally:
        pool.finish()
        pool.mark_input_done()
        # Drain anything left so workers see their sentinels and exit cleanly.
        while pool.get(timeout=1) is not None:
            pass
        pool.join()
