import pytest

from pdf_md.pdf_input import DocInput, InputType, detect_input_type


def test_detect_pdf_file(sample_pdf):
    assert detect_input_type(str(sample_pdf)) == InputType.PDF_FILE


def test_detect_directory(sample_pdf_dir):
    assert detect_input_type(str(sample_pdf_dir)) == InputType.DIRECTORY


def test_detect_hf_prefix():
    assert detect_input_type("hf://org/dataset") == InputType.HF_DATASET


def test_detect_hf_shaped_string():
    assert detect_input_type("org/some-dataset") == InputType.HF_DATASET


def test_detect_unknown_raises():
    with pytest.raises(ValueError):
        detect_input_type("no-such-thing.txt")


def test_docinput_dataclass():
    d = DocInput(doc_id="a", source="a.pdf", path="a.pdf", data=None)
    assert d.doc_id == "a"


from pdf_md.pdf_input import load_docs


def test_load_docs_single_file(sample_pdf):
    docs = list(load_docs(str(sample_pdf)))
    assert len(docs) == 1
    assert docs[0].path == str(sample_pdf)
    assert docs[0].doc_id == "sample"


def test_load_docs_directory(sample_pdf_dir):
    docs = list(load_docs(str(sample_pdf_dir)))
    ids = sorted(d.doc_id for d in docs)
    assert ids == ["doc_a", "doc_b"]
    assert all(d.path is not None for d in docs)


def test_load_docs_resume_skips_completed(sample_pdf_dir):
    docs = list(load_docs(str(sample_pdf_dir), completed_docs={"doc_a"}))
    assert [d.doc_id for d in docs] == ["doc_b"]


from pdf_md.pdf_input import _bounded_download


def test_bounded_download_preserves_order_and_completeness():
    files = [f"f{i}.pdf" for i in range(20)]
    out = list(_bounded_download(files, lambda n: n.upper(), max_workers=4))
    assert out == [f.upper() for f in files]


def test_bounded_download_handles_fewer_files_than_workers():
    out = list(_bounded_download(["a.pdf", "b.pdf"], lambda n: n, max_workers=8))
    assert out == ["a.pdf", "b.pdf"]


def test_bounded_download_does_not_fetch_whole_repo_on_early_stop():
    """A consumer that stops early (max_docs) must not trigger a download of
    every remaining file — only a bounded look-ahead window."""
    downloaded = []
    files = [f"f{i}.pdf" for i in range(1000)]

    def fake_dl(name):
        downloaded.append(name)
        return name

    gen = _bounded_download(files, fake_dl, max_workers=4)
    first_ten = [next(gen) for _ in range(10)]
    gen.close()

    assert first_ten == files[:10]
    # 10 consumed + at most `max_workers` look-ahead — nowhere near 1000
    assert len(downloaded) <= 10 + 4


def test_load_docs_passes_download_workers_to_hf_repo(monkeypatch):
    """download_workers must reach the HF file-repo download look-ahead."""
    captured = {}

    def fake_repo_files(repo_id, token=None, download_workers=None):
        captured["download_workers"] = download_workers
        return iter([])

    monkeypatch.setattr("pdf_md.pdf_input._load_hf_repo_files", fake_repo_files)
    list(load_docs("hf://org/repo", download_workers=23))
    assert captured["download_workers"] == 23


def test_load_docs_hf_dataset_bytes(monkeypatch, sample_pdf):
    pdf_bytes = sample_pdf.read_bytes()
    fake_rows = [
        {"id": "row0", "pdf": pdf_bytes},
        {"id": "row1", "pdf": {"bytes": pdf_bytes}},
    ]

    class _FakeDS:
        column_names = ["id", "pdf"]
        def __iter__(self):
            return iter(fake_rows)

    monkeypatch.setattr(
        "pdf_md.pdf_input._load_hf_repo_files",
        lambda *a, **k: (_ for _ in ()).throw(Exception("not a file repo")),
    )
    monkeypatch.setattr(
        "datasets.load_dataset", lambda *a, **k: _FakeDS()
    )
    docs = list(load_docs("org/fake-dataset"))
    assert [d.doc_id for d in docs] == ["row0", "row1"]
    assert all(d.data == pdf_bytes for d in docs)
