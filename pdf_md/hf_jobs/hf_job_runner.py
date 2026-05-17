# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "pdf-md @ https://github.com/cometadata/pdf-md/archive/refs/heads/main.tar.gz",
#     "pymupdf4llm",
#     "huggingface-hub",
#     "datasets>=4.0.0",
#     "pyarrow>=12.0.0",
#     "requests",
# ]
# ///

from __future__ import annotations

import logging
import os
import signal
import sys
import time
from pathlib import Path

LOGGER = logging.getLogger(__name__)


def _log_memory() -> None:
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    LOGGER.info("Memory: RSS = %.1f MB", int(line.split()[1]) / 1024)
                    return
    except Exception:
        pass


def _bool_env(name: str) -> bool:
    return os.environ.get(name, "false").lower() in {"true", "1", "yes"}


def _int_env(name: str):
    val = os.environ.get(name)
    return int(val) if val else None


def _job_config_from_env() -> dict:
    source = os.environ.get("INPUT_SOURCE")
    if not source:
        raise RuntimeError("INPUT_SOURCE environment variable must be set")
    return {
        "source": source,
        "format": os.environ.get("FORMAT", "markdown"),
        "layout_feature_set": os.environ.get("LAYOUT_FEATURE_SET", "rf"),
        "hf_repo_id": os.environ.get("HF_REPO_ID"),
        "private": _bool_env("PRIVATE"),
        "workers": _int_env("WORKERS"),
        "max_docs": _int_env("MAX_DOCS"),
        "flush_every": _int_env("FLUSH_EVERY"),
        "no_resume": _bool_env("NO_RESUME"),
        "output_dir": os.environ.get("OUTPUT_DIR", "./outputs"),
        "pdf_column": os.environ.get("PDF_COLUMN"),
        "split": os.environ.get("SPLIT", "train"),
        "use_ocr": not _bool_env("NO_OCR"),
        "force_ocr": _bool_env("FORCE_OCR"),
        "ocr_language": os.environ.get("OCR_LANGUAGE"),
        "doc_timeout": _int_env("DOC_TIMEOUT"),
    }


def ensure_code_checkout() -> Path:
    repo_id = os.environ.get("JOB_CODE_REPO")
    if not repo_id:
        return Path(".")
    from huggingface_hub import snapshot_download

    repo_type = os.environ.get("JOB_CODE_REPO_TYPE", "dataset")
    revision = os.environ.get("JOB_CODE_REVISION")
    local_dir = Path(os.environ.get("JOB_CODE_LOCAL_DIR", "/tmp/pdf-md-job-code"))
    local_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo_id, repo_type=repo_type, revision=revision,
        local_dir=str(local_dir),
    )
    return local_dir


def main() -> None:
    import faulthandler
    faulthandler.enable()

    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    def _sigterm(signum, frame):
        LOGGER.info("Received SIGTERM, shutting down")
        raise SystemExit(1)

    signal.signal(signal.SIGTERM, _sigterm)

    code_dir = ensure_code_checkout()
    if str(code_dir) not in sys.path:
        sys.path.insert(0, str(code_dir))

    from pdf_md import diagnostics
    from pdf_md.config import ExtractConfig
    from pdf_md.convert import convert_docs_streaming
    from pdf_md.pdf_input import load_docs
    from pdf_md.storage import (
        AsyncShardUploader, save_batch_incremental, load_hub_progress,
    )

    job = _job_config_from_env()
    hf_token = os.environ.get("HF_TOKEN")

    LOGGER.info(
        "Starting pdf_md job: source=%s format=%s workers=%s repo=%s "
        "use_ocr=%s layout_feature_set=%s",
        job["source"], job["format"], job["workers"], job["hf_repo_id"],
        job["use_ocr"], job["layout_feature_set"],
    )

    config = ExtractConfig().with_overrides(
        format=job["format"], workers=job["workers"],
        flush_every=job["flush_every"], use_ocr=job["use_ocr"],
        force_ocr=job["force_ocr"], ocr_language=job["ocr_language"],
        layout_feature_set=job["layout_feature_set"],
        doc_timeout=job["doc_timeout"],
    )

    shard_index = 0
    completed_docs: set = set()
    if job["hf_repo_id"] and not job["no_resume"]:
        shard_index, completed_docs = load_hub_progress(job["hf_repo_id"], token=hf_token)
    elif job["hf_repo_id"] and job["no_resume"]:
        LOGGER.info("Resume disabled (NO_RESUME); starting fresh")

    docs = load_docs(
        job["source"], pdf_column=job["pdf_column"], split=job["split"],
        token=hf_token, completed_docs=completed_docs,
        download_workers=config.effective_download_workers(),
    )

    flush_every = config.flush_every
    batch_count = 0
    total_docs = 0
    pending_hub_rows: list = []
    job_start = time.monotonic()

    uploader = None
    if job["hf_repo_id"]:
        uploader = AsyncShardUploader(
            job["hf_repo_id"], fmt=config.format, token=hf_token,
            private=job["private"],
        )
        uploader.start()

    try:
        for batch in convert_docs_streaming(docs, config, max_docs=job["max_docs"]):
            batch_count += 1
            total_docs += len({row[0] for row in batch})
            _log_memory()

            if uploader is not None:
                pending_hub_rows.extend(batch)
                if batch_count % flush_every == 0 and pending_hub_rows:
                    uploader.submit(pending_hub_rows, shard_index)
                    shard_index += 1
                    pending_hub_rows = []
            else:
                with diagnostics.timed("runner.save_batch_incremental"):
                    save_batch_incremental(
                        batch, Path(job["output_dir"]), fmt=config.format,
                    )
    finally:
        if uploader is not None:
            if pending_hub_rows:
                LOGGER.info("Flushing %d pending rows on shutdown", len(pending_hub_rows))
                uploader.submit(pending_hub_rows, shard_index)
            try:
                uploader.close()
            except Exception:
                LOGGER.exception("Failed to flush pending shards on shutdown")

    elapsed = time.monotonic() - job_start
    LOGGER.info(
        "Performance summary: %d docs | %.1fs total | %.2f docs/s | workers=%s",
        total_docs, elapsed,
        total_docs / elapsed if elapsed > 0 else 0,
        config.effective_workers(),
    )


if __name__ == "__main__":
    main()
