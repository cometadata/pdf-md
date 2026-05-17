from __future__ import annotations

import json
import logging
import re

LOGGER = logging.getLogger(__name__)

# pymupdf4llm can emit unpaired UTF-16 surrogate code points (e.g. '\ud835'
# from math-heavy text). They break both parquet serialisation and utf-8 file
# writes, so they must be scrubbed before a PageResult leaves the worker.
_SURROGATE_RE = re.compile(r"[\ud800-\udfff]")


def _sanitize_surrogates(text: str) -> str:
    if not text:
        return text
    try:
        text.encode("utf-8")
        return text
    except UnicodeEncodeError:
        return _SURROGATE_RE.sub("�", text)

# pymupdf-layout (a hard dependency of pymupdf4llm, auto-used by pymupdf)
# creates its onnxruntime InferenceSessions with no SessionOptions, so
# onnxruntime sizes its intra-op thread pool to the host core count. With one
# worker process per CPU that means N_cpu^2 onnx threads thrashing N_cpu cores.
# Parallelism here is at the *process* level, so each worker's onnx sessions
# only need a small thread pool.
_ONNX_INTRA_OP_THREADS = 1


def _capped_session_kwargs(args: tuple, kwargs: dict, max_threads: int) -> dict:
    if kwargs.get("sess_options") is not None or len(args) >= 2:
        return kwargs
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = max_threads
    opts.inter_op_num_threads = 1
    return {**kwargs, "sess_options": opts}


def _cap_onnxruntime_threads(max_threads: int = _ONNX_INTRA_OP_THREADS) -> None:
    try:
        import onnxruntime as ort
    except ImportError:  # pragma: no cover - onnxruntime always present in prod
        return
    if getattr(ort.InferenceSession, "_pdf_md_thread_capped", False):
        return

    _orig = ort.InferenceSession

    def _capped(*args, **kwargs):
        return _orig(*args, **_capped_session_kwargs(args, kwargs, max_threads))

    _capped._pdf_md_thread_capped = True
    ort.InferenceSession = _capped


_cap_onnxruntime_threads()

# Install perf instrumentation (no-op unless PDF_MD_DIAG is set). Must run
# before pymupdf4llm is imported: it patches onnxruntime's session constructor,
# and importing the layout stack builds the sessions we want to instrument.
from . import diagnostics  # noqa: E402

diagnostics.install()

import pymupdf  # noqa: E402
import pymupdf4llm  # noqa: E402

from .config import ExtractConfig  # noqa: E402
from .pdf_input import DocInput  # noqa: E402
from .types import ConversionResult, PageResult  # noqa: E402


# What ``pymupdf.layout.activate()`` builds by default. Tracked per worker
# process so a feature-set swap happens at most once per process.
_LAYOUT_FEATURE_SET = "imf+rf"


def _ensure_layout_feature_set(name: str) -> None:
    global _LAYOUT_FEATURE_SET
    if name == _LAYOUT_FEATURE_SET:
        return
    from pymupdf.layout.DocumentLayoutAnalyzer import get_model

    model = get_model(feature_set_name=name)
    pymupdf._get_layout = lambda *args, **kwargs: model.predict(args[0])
    _LAYOUT_FEATURE_SET = name
    LOGGER.info("Layout model feature set set to %r", name)


def _open(doc: DocInput) -> "pymupdf.Document":
    if doc.path is not None:
        return pymupdf.open(doc.path)
    if doc.data is not None:
        return pymupdf.open(stream=doc.data, filetype="pdf")
    raise ValueError(f"DocInput {doc.doc_id} has neither path nor data")


def _ocr_kwargs(config: ExtractConfig) -> dict:
    return {
        "use_ocr": config.use_ocr,
        "force_ocr": config.force_ocr,
        "ocr_language": config.ocr_language,
        "ocr_dpi": config.ocr_dpi,
    }


def _image_kwargs(config: ExtractConfig) -> dict:
    return {
        "write_images": config.write_images,
        "image_path": config.image_path or "",
        "image_format": config.image_format,
        "dpi": config.dpi,
    }


def extract_doc(doc: DocInput, config: ExtractConfig) -> ConversionResult:
    _ensure_layout_feature_set(config.layout_feature_set)
    pdf = None
    try:
        pdf = _open(doc)
        fmt = config.format

        if fmt in ("markdown", "chunks"):
            chunks = pymupdf4llm.to_markdown(
                pdf, page_chunks=True,
                **_ocr_kwargs(config), **_image_kwargs(config),
            )
            pages = []
            for ch in chunks:
                page_index = ch["metadata"].get("page_number", 0)
                content = ch["text"] if fmt == "markdown" else json.dumps(ch, default=str)
                pages.append(PageResult(page_index=page_index, content=content))

        elif fmt == "text":
            chunks = pymupdf4llm.to_text(
                pdf, page_chunks=True, **_ocr_kwargs(config),
            )
            pages = [
                PageResult(
                    page_index=ch["metadata"].get("page_number", 0),
                    content=ch["text"],
                )
                for ch in chunks
            ]

        elif fmt == "json":
            js = pymupdf4llm.to_json(pdf, **_ocr_kwargs(config))
            pages = [PageResult(page_index=0, content=js)]

        else:
            raise ValueError(f"Unknown format: {fmt!r}")

        for page in pages:
            page.content = _sanitize_surrogates(page.content)
        pages.sort(key=lambda p: p.page_index)
        return ConversionResult(doc_id=doc.doc_id, source=doc.source, pages=pages)

    except Exception as exc:
        LOGGER.warning("Failed to extract %s: %s", doc.doc_id, exc, exc_info=True)
        return ConversionResult(
            doc_id=doc.doc_id, source=doc.source, pages=[], error=repr(exc),
        )
    finally:
        if pdf is not None:
            pdf.close()
