from __future__ import annotations

import json
import logging
import queue
import shutil
import threading
import time
from pathlib import Path
from typing import List, Optional, Tuple

from . import diagnostics
from .types import PageResult

LOGGER = logging.getLogger(__name__)

CHECKPOINT_DIR_NAME = ".checkpoints"
PAGE_SEPARATOR = "\n\n<!-- page {n} -->\n\n"

Row = Tuple[str, str, PageResult, Optional[str]]

_EXT = {"markdown": ".md", "text": ".txt", "json": ".json", "chunks": ".jsonl"}


def save_batch_incremental(batch: List[Row], output_dir: Path, fmt: str = "markdown") -> None:
    output_dir = Path(output_dir)
    from collections import defaultdict
    by_doc: dict = defaultdict(list)
    for doc_id, source, page_result, error in batch:
        by_doc[doc_id].append((page_result, error))

    ext = _EXT.get(fmt, ".md")
    for doc_id, pages in by_doc.items():
        out_path = output_dir / f"{doc_id}{ext}"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        existing = out_path.exists() and out_path.stat().st_size > 0
        parts: List[str] = []
        for page_result, error in pages:
            if error is not None:
                continue
            if fmt in ("json", "chunks"):
                parts.append(page_result.content.rstrip() + "\n")
            else:
                if existing or parts:
                    parts.append(PAGE_SEPARATOR.format(n=page_result.page_index))
                parts.append(page_result.content)
        if parts:
            with open(out_path, "a", encoding="utf-8") as f:
                f.write("".join(parts))


def save_batch_checkpoint(batch: List[Row], output_dir: Path, batch_index: int) -> None:
    ckpt_dir = Path(output_dir) / CHECKPOINT_DIR_NAME
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "doc_id": doc_id, "source": source,
            "page_index": pr.page_index, "content": pr.content, "error": error,
        }
        for doc_id, source, pr, error in batch
    ]
    (ckpt_dir / f"batch_{batch_index:05d}.json").write_text(
        json.dumps(rows), encoding="utf-8",
    )


def load_checkpoints(output_dir: Path) -> List[Row]:
    ckpt_dir = Path(output_dir) / CHECKPOINT_DIR_NAME
    if not ckpt_dir.is_dir():
        return []
    rows: List[Row] = []
    for path in sorted(ckpt_dir.glob("batch_*.json")):
        for r in json.loads(path.read_text(encoding="utf-8")):
            rows.append((
                r["doc_id"], r["source"],
                PageResult(page_index=r["page_index"], content=r["content"]),
                r["error"],
            ))
    LOGGER.info("Loaded %d checkpointed rows", len(rows))
    return rows


def clear_checkpoints(output_dir: Path) -> None:
    ckpt_dir = Path(output_dir) / CHECKPOINT_DIR_NAME
    if ckpt_dir.is_dir():
        shutil.rmtree(ckpt_dir)


def completed_docs_from_checkpoints(output_dir: Path) -> set:
    return {row[0] for row in load_checkpoints(output_dir)}


try:
    from huggingface_hub import HfApi, hf_hub_download
except ImportError:  # pragma: no cover
    HfApi = None
    hf_hub_download = None


def _retry_with_backoff(fn, max_attempts=3, base_delay=2.0):
    for attempt in range(1, max_attempts + 1):
        try:
            return fn()
        except Exception:
            if attempt == max_attempts:
                raise
            time.sleep(base_delay * (2 ** (attempt - 1)))


def push_batch_to_hub(
    batch: List[Row],
    repo_id: str,
    shard_index: int,
    *,
    fmt: str = "markdown",
    token: Optional[str] = None,
    private: bool = False,
) -> None:
    import tempfile
    import pyarrow as pa
    import pyarrow.parquet as pq

    rows = [
        {
            "doc_id": doc_id, "source": source,
            "page_index": pr.page_index, "content": pr.content,
            "error": error, "format": fmt,
        }
        for doc_id, source, pr, error in batch
    ]
    table = pa.Table.from_pylist(rows)
    shard_name = f"data/shard_{shard_index:05d}.parquet"

    api = HfApi(token=token)
    api.create_repo(repo_id, repo_type="dataset", private=private, exist_ok=True)

    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=True) as tmp:
        pq.write_table(table, tmp.name)

        def _upload():
            api.upload_file(
                path_or_fileobj=tmp.name, path_in_repo=shard_name,
                repo_id=repo_id, repo_type="dataset",
                commit_message=f"Add shard {shard_index:05d}",
            )
        _retry_with_backoff(_upload)
    LOGGER.info("Pushed %s (%d rows) to %s", shard_name, len(rows), repo_id)


class AsyncShardUploader:

    def __init__(
        self,
        repo_id: str,
        *,
        fmt: str = "markdown",
        token: Optional[str] = None,
        private: bool = False,
        max_pending: int = 8,
    ) -> None:
        self._repo_id = repo_id
        self._fmt = fmt
        self._token = token
        self._private = private
        self._queue: "queue.Queue" = queue.Queue(maxsize=max_pending)
        self._thread = threading.Thread(
            target=self._run, name="shard-uploader", daemon=True,
        )
        self._error: Optional[BaseException] = None
        self._started = False

    def start(self) -> None:
        self._started = True
        self._thread.start()

    def submit(self, rows: List[Row], shard_index: int) -> None:
        self._queue.put((list(rows), shard_index))

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                return
            rows, shard_index = item
            try:
                with diagnostics.timed("uploader.push_batch_to_hub"):
                    push_batch_to_hub(
                        rows, self._repo_id, shard_index, fmt=self._fmt,
                        token=self._token, private=self._private,
                    )
            except Exception as exc:  # noqa: BLE001 - retained, re-raised by close()
                if self._error is None:
                    self._error = exc
                LOGGER.exception(
                    "Async shard upload failed (shard %d)", shard_index,
                )

    def close(self, timeout: float = 600.0) -> None:
        if not self._started:
            return
        self._queue.put(None)
        self._thread.join(timeout)
        if self._error is not None:
            raise self._error


import re

_SHARD_RE = re.compile(r"^data/shard_(\d+)\.parquet$")


def load_hub_progress(
    repo_id: str, token: Optional[str] = None,
) -> Tuple[int, set]:
    import pyarrow.parquet as pq

    try:
        api = HfApi(token=token)
        files = api.list_repo_files(repo_id, repo_type="dataset")
    except Exception:
        LOGGER.info("No existing repo %s; starting fresh", repo_id)
        return 0, set()

    shard_files = []
    max_index = -1
    for f in files:
        m = _SHARD_RE.match(f)
        if m:
            shard_files.append(f)
            max_index = max(max_index, int(m.group(1)))
    if not shard_files:
        return 0, set()

    completed: set = set()
    for sf in shard_files:
        try:
            local = hf_hub_download(repo_id, sf, repo_type="dataset", token=token)
            table = pq.read_table(local, columns=["doc_id"])
            completed.update(table.column("doc_id").to_pylist())
        except Exception:
            LOGGER.warning("Failed to read shard %s; skipping", sf)

    LOGGER.info(
        "Resume: %d shards, %d completed docs, next shard=%d",
        len(shard_files), len(completed), max_index + 1,
    )
    return max_index + 1, completed
