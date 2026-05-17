from __future__ import annotations

import argparse
import logging
import sys
from typing import List, Optional


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="pdf_md",
        description="Fast, parallel PDF to full-text conversion via pymupdf4llm.",
    )
    p.add_argument("source", help="PDF file, directory, or HF dataset reference")
    p.add_argument("--format", default="markdown",
                   choices=["markdown", "text", "json", "chunks"],
                   help="Output format (default: markdown)")
    p.add_argument("--output", "-o", default=None,
                   help="Local output directory or HF repo id")
    p.add_argument("--private", action="store_true",
                   help="Create a private HF dataset (with HF-repo --output)")
    p.add_argument("--workers", type=int, default=None,
                   help="Worker process count (default: os.cpu_count())")
    p.add_argument("--layout-feature-set", default="rf",
                   choices=["imf+rf", "rf"],
                   help="Layout model feature set: rf (default, text-only, "
                        "faster) or imf+rf (adds the per-page image CNN)")
    p.add_argument("--max-docs", type=int, default=None,
                   help="Limit total documents processed")
    p.add_argument("--flush-every", type=int, default=None,
                   help="Batches per HF shard flush")
    p.add_argument("--no-ocr", dest="use_ocr", action="store_false",
                   help="Disable OCR (on by default)")
    p.add_argument("--force-ocr", action="store_true",
                   help="Force OCR on every page")
    p.add_argument("--ocr-language", default="eng",
                   help="Tesseract language(s), e.g. eng+fra")
    p.add_argument("--pdf-column", default=None,
                   help="PDF column name for HF datasets")
    p.add_argument("--split", default="train",
                   help="HF dataset split (default: train)")
    p.add_argument("--no-resume", action="store_true",
                   help="Ignore existing progress and start fresh")
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    p.set_defaults(use_ocr=True)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    import pdf_md

    results = pdf_md.convert(
        args.source,
        format=args.format,
        output=args.output,
        private=args.private,
        workers=args.workers,
        layout_feature_set=args.layout_feature_set,
        max_docs=args.max_docs,
        flush_every=args.flush_every,
        use_ocr=args.use_ocr,
        force_ocr=args.force_ocr,
        ocr_language=args.ocr_language,
        pdf_column=args.pdf_column,
        split=args.split,
        no_resume=args.no_resume,
    )

    if args.output is None:
        for result in results:
            if result.error:
                print(f"<!-- {result.doc_id}: ERROR {result.error} -->")
                continue
            for page in result.pages:
                print(page.content)
    else:
        total_pages = sum(len(r.pages) for r in results)
        errors = sum(1 for r in results if r.error)
        print(f"Done: {len(results)} docs, {total_pages} pages, {errors} errors")
    return 0


if __name__ == "__main__":
    sys.exit(main())
