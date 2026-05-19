from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List, Optional

from .config import ExtractConfig
from .pdf_input import load_docs
from .types import ConversionResult, PageResult

# Import the .convert and .storage submodules at package-import time so their
# attribute bindings on `pdf_md` happen *before* `def convert` below. Otherwise
# the lazy `from .convert import ...` inside convert() would bind the
# `pdf_md.convert` submodule over the `convert` function on first call, and a
# second call to `pdf_md.convert(...)` would raise "'module' object is not
# callable". This mirrors pdf_ocr/__init__.py, whose top-level
# `from pdf_ocr.convert import ...` imports the submodule before `def convert`.
from . import convert as _convert_module  # noqa: F401
from . import storage as _storage_module  # noqa: F401

LOGGER = logging.getLogger(__name__)

__all__ = ["convert", "ExtractConfig", "ConversionResult", "PageResult"]


def _is_local_output(output: str) -> bool:
    if output.startswith(("./", "/", "..")):
        return True
    p = Path(output)
    if p.exists():
        return True
    # 'org/name' with no leading dot/slash and not existing -> HF repo
    return not ("/" in output and output.count("/") == 1)


def convert(
    source: str,
    *,
    format: str = "markdown",
    output: Optional[str] = None,
    private: bool = False,
    workers: Optional[int] = None,
    layout_feature_set: str = "rf",
    max_docs: Optional[int] = None,
    flush_every: Optional[int] = None,
    use_ocr: bool = True,
    force_ocr: bool = False,
    ocr_language: str = "eng",
    pdf_column: Optional[str] = None,
    split: str = "train",
    token: Optional[str] = None,
    no_resume: bool = False,
    parquet: bool = False,
) -> List[ConversionResult]:
    from .convert import convert_docs_streaming, _group_by_document
    from .storage import (
        save_batch_incremental, save_batch_checkpoint, load_checkpoints,
        clear_checkpoints, completed_docs_from_checkpoints,
        AsyncShardUploader, load_hub_progress,
        write_local_shard, load_local_parquet_progress,
    )

    config = ExtractConfig().with_overrides(
        format=format, workers=workers, flush_every=flush_every,
        use_ocr=use_ocr, force_ocr=force_ocr, ocr_language=ocr_language,
        layout_feature_set=layout_feature_set,
    )
    hf_token = token or os.environ.get("HF_TOKEN")

    is_local_output = output is not None and _is_local_output(output)
    is_local_parquet = is_local_output and parquet
    checkpoint_dir = (
        Path(output) if (output and is_local_output and not parquet) else None
    )

    completed_docs: set = set()
    shard_index = 0
    previous_rows: List = []
    if output and not no_resume:
        if is_local_parquet:
            shard_index, completed_docs = load_local_parquet_progress(Path(output))
        elif is_local_output:
            previous_rows = load_checkpoints(checkpoint_dir)
            completed_docs = {r[0] for r in previous_rows}
        else:
            shard_index, completed_docs = load_hub_progress(output, token=hf_token)

    docs = load_docs(
        source, pdf_column=pdf_column, split=split, token=hf_token,
        completed_docs=completed_docs,
        download_workers=config.effective_download_workers(),
    )

    all_rows: List = list(previous_rows)
    batch_count = 0
    pending_hub_rows: List = []
    pending_local_rows: List = []

    uploader = None
    if output and not is_local_output:
        uploader = AsyncShardUploader(
            output, fmt=config.format, token=hf_token, private=private,
        )
        uploader.start()

    try:
        for batch in convert_docs_streaming(docs, config, max_docs=max_docs):
            all_rows.extend(batch)
            batch_count += 1

            if is_local_parquet:
                pending_local_rows.extend(batch)
                if batch_count % config.flush_every == 0 and pending_local_rows:
                    write_local_shard(
                        pending_local_rows, Path(output), shard_index,
                        fmt=config.format,
                    )
                    shard_index += 1
                    pending_local_rows = []
            elif output and is_local_output:
                save_batch_incremental(batch, checkpoint_dir, fmt=config.format)
                save_batch_checkpoint(batch, checkpoint_dir, batch_count - 1)
            elif uploader is not None:
                pending_hub_rows.extend(batch)
                if batch_count % config.flush_every == 0 and pending_hub_rows:
                    uploader.submit(pending_hub_rows, shard_index)
                    shard_index += 1
                    pending_hub_rows = []
    finally:
        if pending_local_rows:
            write_local_shard(
                pending_local_rows, Path(output), shard_index,
                fmt=config.format,
            )
        if uploader is not None:
            if pending_hub_rows:
                uploader.submit(pending_hub_rows, shard_index)
            uploader.close()

    if checkpoint_dir is not None:
        clear_checkpoints(checkpoint_dir)

    return _group_by_document(all_rows)
