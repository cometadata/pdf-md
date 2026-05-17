from __future__ import annotations

import enum
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional, Set

LOGGER = logging.getLogger(__name__)

# Look-ahead used when load_docs is called without a config-derived value
# (e.g. the library API on a bare HF repo). Callers that have an ExtractConfig
# should pass config.effective_download_workers() instead.
_DEFAULT_DOWNLOAD_WORKERS = 8


class InputType(enum.Enum):
    PDF_FILE = "pdf_file"
    DIRECTORY = "directory"
    HF_DATASET = "hf_dataset"


@dataclass
class DocInput:
    doc_id: str
    source: str
    path: Optional[str] = None     # local filesystem path if available
    data: Optional[bytes] = None   # raw PDF bytes if streamed


def detect_input_type(source: str) -> InputType:
    if source.startswith("hf://"):
        return InputType.HF_DATASET
    path = Path(source)
    if path.is_file() and path.suffix.lower() == ".pdf":
        return InputType.PDF_FILE
    if path.is_dir():
        return InputType.DIRECTORY
    if "/" in source and not path.exists():
        return InputType.HF_DATASET
    raise ValueError(f"Cannot determine input type for: {source!r}")


def _iter_file(source: str) -> Iterator[DocInput]:
    p = Path(source)
    yield DocInput(doc_id=p.stem, source=str(p), path=str(p))


def _iter_directory(source: str) -> Iterator[DocInput]:
    base = Path(source)
    pdf_files = sorted(base.rglob("*.pdf"))
    LOGGER.info("Found %d PDFs in %s", len(pdf_files), source)
    for p in pdf_files:
        doc_id = p.relative_to(base).with_suffix("").as_posix()
        yield DocInput(doc_id=doc_id, source=str(p), path=str(p))


def load_docs(
    source: str,
    *,
    pdf_column: Optional[str] = None,
    split: str = "train",
    token: Optional[str] = None,
    completed_docs: Optional[Set[str]] = None,
    download_workers: int = _DEFAULT_DOWNLOAD_WORKERS,
) -> Iterator[DocInput]:
    completed = completed_docs or set()
    input_type = detect_input_type(source)
    LOGGER.info("Detected input type: %s for source: %s", input_type.value, source)

    if input_type == InputType.PDF_FILE:
        raw = _iter_file(source)
    elif input_type == InputType.DIRECTORY:
        raw = _iter_directory(source)
    else:
        raw = _iter_hf_dataset(
            source, pdf_column=pdf_column, split=split, token=token,
            download_workers=download_workers,
        )

    skipped = 0
    for doc in raw:
        if doc.doc_id in completed:
            skipped += 1
            continue
        yield doc
    if skipped:
        LOGGER.info("Resume: skipped %d already-completed docs", skipped)


from collections import deque
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Iterable, TypeVar

try:
    from huggingface_hub import HfApi, hf_hub_download
except ImportError:  # pragma: no cover
    HfApi = None
    hf_hub_download = None

_T = TypeVar("_T")
_R = TypeVar("_R")


def _bounded_download(
    items: Iterable[_T],
    download_fn: Callable[[_T], _R],
    *,
    max_workers: int = 4,
) -> Iterator[_R]:
    it = iter(items)
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        window: "deque" = deque()
        for _ in range(max_workers):
            try:
                window.append(pool.submit(download_fn, next(it)))
            except StopIteration:
                break
        while window:
            result = window.popleft().result()
            try:
                window.append(pool.submit(download_fn, next(it)))
            except StopIteration:
                pass
            yield result

_PDF_COLUMN_CANDIDATES = ["pdf", "file", "content", "data", "document", "pdf_data", "pdf_bytes"]


def _detect_pdf_column(ds, pdf_column: Optional[str]) -> Optional[str]:
    if pdf_column and pdf_column in ds.column_names:
        return pdf_column
    for name in _PDF_COLUMN_CANDIDATES:
        if name in ds.column_names:
            return name
    return None


def _load_hf_repo_files(
    repo_id: str, token: Optional[str],
    download_workers: int = _DEFAULT_DOWNLOAD_WORKERS,
) -> Iterator[DocInput]:
    api = HfApi()
    all_files = api.list_repo_files(repo_id, repo_type="dataset", token=token)
    pdf_files = sorted(f for f in all_files if f.lower().endswith(".pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in repo {repo_id}")
    LOGGER.info("Found %d PDFs in HF repo %s", len(pdf_files), repo_id)

    def _download(filename: str) -> DocInput:
        local = hf_hub_download(repo_id, filename, repo_type="dataset", token=token)
        doc_id = Path(filename).with_suffix("").as_posix()
        return DocInput(doc_id=doc_id, source=f"{repo_id}/{filename}", path=local)

    yield from _bounded_download(pdf_files, _download, max_workers=download_workers)


def _iter_hf_dataset(
    source: str, *, pdf_column: Optional[str], split: str, token: Optional[str],
    download_workers: int = _DEFAULT_DOWNLOAD_WORKERS,
) -> Iterator[DocInput]:
    repo_id = source.removeprefix("hf://")

    try:
        yield from _load_hf_repo_files(
            repo_id, token=token, download_workers=download_workers,
        )
        return
    except Exception:
        LOGGER.info("%s is not a file repo; trying as parquet dataset", repo_id)

    from datasets import load_dataset

    ds = load_dataset(repo_id, split=split, token=token, streaming=True)
    col = _detect_pdf_column(ds, pdf_column)
    if col is None:
        raise ValueError(
            f"No PDF column in dataset {repo_id!r}. Columns: {ds.column_names}. "
            f"Use pdf_column to specify."
        )
    LOGGER.info("Streaming column %r from dataset %s", col, repo_id)

    for idx, row in enumerate(ds):
        value = row[col]
        doc_id = row.get("id", row.get("doc_id", f"doc_{idx:05d}"))
        if isinstance(doc_id, int):
            doc_id = f"doc_{doc_id:05d}"
        doc_id = str(doc_id)

        if isinstance(value, bytes):
            yield DocInput(doc_id=doc_id, source=repo_id, data=value)
        elif isinstance(value, dict) and "bytes" in value:
            yield DocInput(doc_id=doc_id, source=repo_id, data=value["bytes"])
        elif isinstance(value, str) and value.startswith(("http://", "https://")):
            import requests
            resp = requests.get(value, timeout=60)
            resp.raise_for_status()
            yield DocInput(doc_id=doc_id, source=value, data=resp.content)
        elif isinstance(value, str):
            yield DocInput(doc_id=doc_id, source=value, path=value)
        else:
            LOGGER.warning("Skipping row %d: unsupported value type %s", idx, type(value))
