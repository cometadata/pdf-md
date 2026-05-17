import json
from pathlib import Path

import pytest

from pdf_md.types import PageResult
from pdf_md.storage import (
    save_batch_incremental, save_batch_checkpoint,
    load_checkpoints, clear_checkpoints,
)


def _row(doc_id, page_index, content, error=None):
    return (doc_id, f"{doc_id}.pdf", PageResult(page_index, content), error)


def test_save_batch_incremental_markdown(tmp_path):
    batch = [_row("d1", 0, "page zero"), _row("d1", 1, "page one")]
    save_batch_incremental(batch, tmp_path, fmt="markdown")
    text = (tmp_path / "d1.md").read_text()
    assert "page zero" in text and "page one" in text


def test_save_batch_incremental_appends(tmp_path):
    save_batch_incremental([_row("d1", 0, "first")], tmp_path, fmt="markdown")
    save_batch_incremental([_row("d1", 1, "second")], tmp_path, fmt="markdown")
    text = (tmp_path / "d1.md").read_text()
    assert text.index("first") < text.index("second")


def test_checkpoint_roundtrip(tmp_path):
    batch = [_row("d1", 0, "x"), _row("d2", 0, "y", error="boom")]
    save_batch_checkpoint(batch, tmp_path, 0)
    loaded = load_checkpoints(tmp_path)
    assert len(loaded) == 2
    errors = {r[0]: r[3] for r in loaded}
    assert errors["d2"] == "boom"
    clear_checkpoints(tmp_path)
    assert load_checkpoints(tmp_path) == []


def test_push_batch_to_hub_writes_parquet(tmp_path, monkeypatch):
    import pyarrow.parquet as pq
    from pdf_md import storage

    uploaded = {}

    class _FakeApi:
        def __init__(self, *a, **k): pass
        def create_repo(self, *a, **k): pass
        def upload_file(self, path_or_fileobj, path_in_repo, **k):
            uploaded["path_in_repo"] = path_in_repo
            uploaded["table"] = pq.read_table(path_or_fileobj)

    monkeypatch.setattr(storage, "HfApi", _FakeApi)

    batch = [_row("d1", 0, "hello"), _row("d2", -1, "", error="boom")]
    storage.push_batch_to_hub(batch, repo_id="org/out", shard_index=3, fmt="markdown")

    assert uploaded["path_in_repo"] == "data/shard_00003.parquet"
    cols = set(uploaded["table"].column_names)
    assert cols == {"doc_id", "source", "page_index", "content", "error", "format"}


def test_load_hub_progress_returns_completed_doc_ids(tmp_path, monkeypatch):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from pdf_md import storage

    # build two fake shard parquet files on disk
    shard0 = tmp_path / "shard_00000.parquet"
    shard1 = tmp_path / "shard_00001.parquet"
    pq.write_table(pa.table({"doc_id": ["a", "b"], "page_index": [0, 0]}), shard0)
    pq.write_table(pa.table({"doc_id": ["c"], "page_index": [0]}), shard1)

    class _FakeApi:
        def __init__(self, *a, **k): pass
        def list_repo_files(self, *a, **k):
            return ["data/shard_00000.parquet", "data/shard_00001.parquet", "README.md"]

    monkeypatch.setattr(storage, "HfApi", _FakeApi)
    monkeypatch.setattr(
        storage, "hf_hub_download",
        lambda repo_id, f, **k: str(shard0 if "00000" in f else shard1),
    )

    next_index, completed = storage.load_hub_progress("org/out")
    assert next_index == 2
    assert completed == {"a", "b", "c"}


def test_load_hub_progress_missing_repo(monkeypatch):
    from pdf_md import storage

    class _FakeApi:
        def __init__(self, *a, **k): pass
        def list_repo_files(self, *a, **k):
            raise Exception("404")

    monkeypatch.setattr(storage, "HfApi", _FakeApi)
    assert storage.load_hub_progress("org/missing") == (0, set())


def test_async_uploader_uploads_all_shards_with_indices(monkeypatch):
    from pdf_md import storage

    calls = []
    monkeypatch.setattr(
        storage, "push_batch_to_hub",
        lambda rows, repo_id, shard_index, **k: calls.append((shard_index, list(rows))),
    )

    up = storage.AsyncShardUploader("org/out", fmt="markdown")
    up.start()
    up.submit([_row("d1", 0, "a")], 0)
    up.submit([_row("d2", 0, "b")], 1)
    up.submit([_row("d3", 0, "c")], 2)
    up.close()

    assert sorted(c[0] for c in calls) == [0, 1, 2]


def test_async_uploader_reraises_first_error_on_close(monkeypatch):
    from pdf_md import storage

    def boom(*a, **k):
        raise RuntimeError("hub down")

    monkeypatch.setattr(storage, "push_batch_to_hub", boom)

    up = storage.AsyncShardUploader("org/out")
    up.start()
    up.submit([_row("d1", 0, "a")], 0)
    with pytest.raises(RuntimeError, match="hub down"):
        up.close()


def test_async_uploader_close_without_start_is_noop():
    from pdf_md import storage

    # Must return immediately — not block waiting on an unstarted thread.
    storage.AsyncShardUploader("org/out").close()
