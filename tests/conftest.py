import shutil
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
_SRC_PDF = _REPO / "pymupdf4llm" / "tests" / "test_370.pdf"


@pytest.fixture
def sample_pdf(tmp_path):
    """A small real PDF copied into a tmp dir, named sample.pdf."""
    dest = tmp_path / "sample.pdf"
    shutil.copy(_SRC_PDF, dest)
    return dest


@pytest.fixture
def sample_pdf_dir(tmp_path):
    """A directory containing two copies of the sample PDF."""
    d = tmp_path / "pdfs"
    d.mkdir()
    shutil.copy(_SRC_PDF, d / "doc_a.pdf")
    shutil.copy(_SRC_PDF, d / "doc_b.pdf")
    return d
