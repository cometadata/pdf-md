from pathlib import Path

import pdf_md


def test_convert_single_pdf_returns_results(sample_pdf):
    results = pdf_md.convert(str(sample_pdf), use_ocr=False, workers=2)
    assert len(results) == 1
    assert results[0].doc_id == "sample"
    assert len(results[0].pages) >= 1


def test_convert_directory_to_local_output(sample_pdf_dir, tmp_path):
    out = tmp_path / "out"
    results = pdf_md.convert(
        str(sample_pdf_dir), output=str(out), use_ocr=False, workers=2,
    )
    assert sorted(r.doc_id for r in results) == ["doc_a", "doc_b"]
    assert (out / "doc_a.md").exists()
    assert (out / "doc_b.md").exists()
    # checkpoints cleared on success
    assert not (out / ".checkpoints").exists()
