# pdf_md

Fast, parallel PDF to full-text conversion built on the CPU-only [`pymupdf4llm`](https://pymupdf.readthedocs.io/en/latest/pymupdf4llm/) engine.

Accepts a PDF file, a directory of PDFs, or a HuggingFace dataset as input. Outputs full text to local files, stdout, or a HuggingFace dataset repo.

## Usage

### CLI

```bash
# Single PDF to stdout
python -m pdf_md paper.pdf

# Directory of PDFs to local files
python -m pdf_md ./pdfs/ --output ./output/

# Plain text instead of markdown, 16 workers
python -m pdf_md ./pdfs/ --format text --workers 16 --output ./output/

# HuggingFace dataset input, push results to a (private) HF dataset repo
python -m pdf_md hf://org/dataset --output org/output-dataset --private
```

### Python API

```python
from pdf_md import convert

# Single PDF -> list of ConversionResult
results = convert("paper.pdf")
for result in results:
    if result.error:
        continue
    for page in result.pages:
        print(page.content)

# Directory in, local files out
results = convert("./pdfs/", output="./output/")

# HF dataset in, HF dataset repo out
results = convert(
    "hf://org/dataset",
    format="text",
    output="org/output-dataset",
    private=True,
    workers=16,
)
```

`convert()` signature:

```python
def convert(
    source: str,
    *,
    format: str = "markdown",          # markdown | text | json | chunks
    output: Optional[str] = None,      # local dir OR HF repo id (org/name)
    private: bool = False,
    workers: Optional[int] = None,     # None -> os.cpu_count()
    max_docs: Optional[int] = None,
    flush_every: Optional[int] = None,
    use_ocr: bool = True,
    force_ocr: bool = False,
    ocr_language: str = "eng",
    pdf_column: Optional[str] = None,
    split: str = "train",
    token: Optional[str] = None,
    no_resume: bool = False,
) -> List[ConversionResult]:
    ...
```

`convert()` returns a `List[ConversionResult]`, one per input document:

- `ConversionResult` — `doc_id` (str), `source` (str), `pages` (`List[PageResult]`), `error` (`Optional[str]`, `None` on success).
- `PageResult` — `page_index` (int), `content` (str).

## CLI Options

| Flag | Description |
|---|---|
| `source` | PDF file, directory, or HF dataset reference (positional) |
| `--format` | Output format: `markdown` (default), `text`, `json`, `chunks` |
| `--output`, `-o` | Local output directory or HF repo id (default: stdout) |
| `--private` | Create a private HF dataset (with an HF-repo `--output`) |
| `--workers` | Worker process count (default: `os.cpu_count()`) |
| `--layout-feature-set` | Layout model feature set: `rf` (default, text-only) or `imf+rf` (adds the per-page image CNN) |
| `--max-docs` | Limit total documents processed |
| `--flush-every` | Batches per HF shard flush |
| `--no-ocr` | Disable OCR (OCR is on by default) |
| `--force-ocr` | Force OCR on every page |
| `--ocr-language` | Tesseract language(s), e.g. `eng+fra` (default: `eng`) |
| `--pdf-column` | PDF column name for HF datasets |
| `--split` | HF dataset split (default: `train`) |
| `--no-resume` | Ignore existing progress and start fresh |
| `--log-level` | `DEBUG`, `INFO`, `WARNING`, `ERROR` (default: `INFO`) |

## Output

Results are emitted as flat rows. HF parquet shards carry these columns:

| Column | Description |
|---|---|
| `doc_id` | Document identifier (relative path stem for directories/HF file repos) |
| `source` | Original path, URL, or HF repo id |
| `page_index` | Source page number for the row; `-1` for a failed-document row |
| `content` | Markdown / text / JSON string per `--format` (empty on a failed-document row) |
| `error` | `null` for successful pages; a string on a failed-document row |
| `format` | The `--format` used for the run |

- One row per page for `markdown`, `text`, and `chunks`.
- The `json` format emits **one row per document** (`page_index=0`) — `pymupdf4llm.to_json()` produces a single whole-document JSON string with no page chunking.
- A document that fails entirely emits a single row with `page_index=-1`, empty `content`, and `error` set.

Local output (`--output ./dir/`) writes one file per document, named `<doc_id>.<ext>` — `.md` (markdown), `.txt` (text), `.json` (json), `.jsonl` (chunks). Pages are concatenated with `<!-- page N -->` separators for markdown/text and JSON-lines for json/chunks. Per-batch checkpoints are written under `.checkpoints/` and cleared on successful completion.

## HuggingFace Jobs

`pdf_md/hf_jobs/hf_job_runner.py` is an inline-deps (PEP 723) script that runs `pdf_md` as a HuggingFace job: HF dataset in, HF dataset out. It is configured entirely through environment variables.

| Env var | Meaning |
|---|---|
| `INPUT_SOURCE` | Required; PDF file, directory, or HF dataset |
| `FORMAT` | Output format (default: `markdown`) |
| `LAYOUT_FEATURE_SET` | Layout model feature set: `rf` (default, text-only) or `imf+rf` (adds the per-page image CNN) |
| `HF_REPO_ID` | Output dataset repo; if unset, results go to `OUTPUT_DIR` |
| `HF_TOKEN` | HuggingFace auth token |
| `PRIVATE` | `true`/`false` — create a private output dataset |
| `WORKERS` | Worker process count (default: `os.cpu_count()`) |
| `MAX_DOCS` | Cap on total documents processed |
| `FLUSH_EVERY` | Batches per HF shard flush |
| `NO_RESUME` | `true`/`false` — skip resume, start fresh |
| `OUTPUT_DIR` | Local output dir when `HF_REPO_ID` is unset (default: `./outputs`) |
| `PDF_COLUMN` | PDF column name for HF datasets |
| `SPLIT` | HF dataset split (default: `train`) |
| `NO_OCR` | `true`/`false` — disable OCR (OCR is on by default) |
| `FORCE_OCR` | `true`/`false` — force OCR on every page |
| `OCR_LANGUAGE` | Tesseract language(s), e.g. `eng+fra` (default: `eng`) |
| `LOG_LEVEL` | Logging level (default: `INFO`) |
| `JOB_CODE_REPO` / `JOB_CODE_REPO_TYPE` / `JOB_CODE_REVISION` / `JOB_CODE_LOCAL_DIR` | Optional code checkout via `snapshot_download` (`JOB_CODE_REPO_TYPE` default: `dataset`, `JOB_CODE_LOCAL_DIR` default: `/tmp/pdf-md-job-code`) |

Example invocation:

```bash
INPUT_SOURCE="hf://org/input-pdfs" \
HF_REPO_ID="org/output-dataset" \
HF_TOKEN="hf_..." \
FORMAT="markdown" \
WORKERS="16" \
PRIVATE="true" \
python pdf_md/hf_jobs/hf_job_runner.py
```

The runner resumes by reading completed `doc_id`s from existing shards (`load_hub_progress`), streams batches through `convert_docs_streaming`, pushes a parquet shard every `FLUSH_EVERY` batches, flushes any remainder on shutdown, and logs a performance summary (docs, elapsed, docs/s).

## Package Structure

```
pdf-md/                    # repo root
├── pyproject.toml
├── README.md
├── tests/
└── pdf_md/                # the package
    ├── __init__.py        # public convert() API
    ├── __main__.py        # `python -m pdf_md` → cli.main()
    ├── cli.py             # argparse; flags only (no YAML config)
    ├── config.py          # ExtractConfig frozen dataclass + with_overrides()
    ├── types.py           # DocInput / PageResult / ConversionResult dataclasses
    ├── pdf_input.py       # load_docs(): file / dir / HF dataset → DocInput stream;
    │                      #   skips already-completed doc_ids (resume)
    ├── worker.py          # per-doc extraction: runs pymupdf4llm, emits per-page rows;
    │                      #   the function executed inside pool worker processes
    ├── convert.py         # parallel pipeline: worker-pool supervisor, streaming
    │                      #   generator convert_docs_streaming(), crash respawn
    ├── storage.py         # save_batch_incremental / push_batch_to_hub /
    │                      #   load_hub_progress / checkpoints / clear_checkpoints
    └── hf_jobs/
        ├── __init__.py
        └── hf_job_runner.py   # inline-deps script; env-var contract; near-identical
                               #   to pdf_ocr/hf_jobs/hf_job_runner.py minus server
```
