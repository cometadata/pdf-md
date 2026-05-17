from pdf_md.types import ConversionResult, PageResult


def test_dataclasses():
    pr = PageResult(page_index=2, content="hi")
    cr = ConversionResult(doc_id="d", source="s", pages=[pr])
    assert cr.pages[0].content == "hi"
    assert cr.error is None


import json

from pdf_md.config import ExtractConfig
from pdf_md.pdf_input import DocInput
from pdf_md.worker import extract_doc


def test_extract_markdown_from_path(sample_pdf):
    doc = DocInput(doc_id="sample", source=str(sample_pdf), path=str(sample_pdf))
    result = extract_doc(doc, ExtractConfig(format="markdown", use_ocr=False))
    assert result.error is None
    assert len(result.pages) >= 1
    assert all(isinstance(p.content, str) for p in result.pages)
    # page_index values are unique and ordered
    idxs = [p.page_index for p in result.pages]
    assert idxs == sorted(idxs)


def test_extract_text_from_bytes(sample_pdf):
    doc = DocInput(doc_id="sample", source="mem", data=sample_pdf.read_bytes())
    result = extract_doc(doc, ExtractConfig(format="text", use_ocr=False))
    assert result.error is None
    assert len(result.pages) >= 1


def test_extract_json_emits_single_row(sample_pdf):
    doc = DocInput(doc_id="sample", source=str(sample_pdf), path=str(sample_pdf))
    result = extract_doc(doc, ExtractConfig(format="json", use_ocr=False))
    assert result.error is None
    assert len(result.pages) == 1
    assert result.pages[0].page_index == 0
    json.loads(result.pages[0].content)  # must be valid JSON


def test_extract_chunks_serializes_dict(sample_pdf):
    doc = DocInput(doc_id="sample", source=str(sample_pdf), path=str(sample_pdf))
    result = extract_doc(doc, ExtractConfig(format="chunks", use_ocr=False))
    assert result.error is None
    chunk = json.loads(result.pages[0].content)
    assert "text" in chunk and "metadata" in chunk


def test_extract_corrupt_pdf_returns_error(tmp_path):
    bad = tmp_path / "bad.pdf"
    bad.write_bytes(b"%PDF-1.4 not really a pdf")
    doc = DocInput(doc_id="bad", source=str(bad), path=str(bad))
    result = extract_doc(doc, ExtractConfig(use_ocr=False))
    assert result.error is not None
    assert result.pages == []


from pdf_md.worker import _capped_session_kwargs


def test_capped_session_kwargs_injects_options_when_none():
    kwargs = _capped_session_kwargs(("model.onnx",), {}, max_threads=2)
    opts = kwargs["sess_options"]
    assert opts.intra_op_num_threads == 2
    assert opts.inter_op_num_threads == 1


def test_capped_session_kwargs_respects_explicit_keyword_options():
    import onnxruntime as ort
    explicit = ort.SessionOptions()
    explicit.intra_op_num_threads = 16
    kwargs = _capped_session_kwargs(
        ("model.onnx",), {"sess_options": explicit}, max_threads=2,
    )
    assert kwargs["sess_options"] is explicit  # untouched


def test_capped_session_kwargs_respects_explicit_positional_options():
    import onnxruntime as ort
    explicit = ort.SessionOptions()
    # sess_options passed positionally (2nd arg) -> must not be overridden
    kwargs = _capped_session_kwargs(
        ("model.onnx", explicit), {}, max_threads=2,
    )
    assert "sess_options" not in kwargs


from pdf_md.worker import _sanitize_surrogates


def test_sanitize_surrogates_leaves_clean_text_untouched():
    s = "Normal text with math αβγ and emoji 🎉"
    assert _sanitize_surrogates(s) is s  # fast path: same object back


def test_sanitize_surrogates_replaces_lone_surrogate():
    s = "before \ud835 after"  # lone high surrogate, breaks utf-8
    out = _sanitize_surrogates(s)
    assert "\ud835" not in out
    assert out == "before � after"
    out.encode("utf-8")  # must now be utf-8 encodable


def test_sanitize_surrogates_empty():
    assert _sanitize_surrogates("") == ""


def test_ensure_layout_feature_set_swaps_and_is_idempotent(monkeypatch):
    import pymupdf
    from pdf_md import worker

    # _ensure_layout_feature_set mutates process-global layout state; save and
    # restore it so the real-extraction tests above keep their imf+rf model.
    orig_flag = worker._LAYOUT_FEATURE_SET
    orig_get_layout = getattr(pymupdf, "_get_layout", None)
    try:
        calls = []

        class _FakeModel:
            def predict(self, page):
                return ["fake-layout"]

        def fake_get_model(feature_set_name):
            calls.append(feature_set_name)
            return _FakeModel()

        monkeypatch.setattr(
            "pymupdf.layout.DocumentLayoutAnalyzer.get_model", fake_get_model
        )

        # no-op when the requested set is already active
        worker._LAYOUT_FEATURE_SET = "imf+rf"
        worker._ensure_layout_feature_set("imf+rf")
        assert calls == []

        # swapping rebuilds the model and rebinds pymupdf._get_layout
        worker._ensure_layout_feature_set("rf")
        assert calls == ["rf"]
        assert worker._LAYOUT_FEATURE_SET == "rf"
        assert pymupdf._get_layout(object()) == ["fake-layout"]

        # idempotent: a second call with the same set rebuilds nothing
        worker._ensure_layout_feature_set("rf")
        assert calls == ["rf"]
    finally:
        worker._LAYOUT_FEATURE_SET = orig_flag
        if orig_get_layout is not None:
            pymupdf._get_layout = orig_get_layout
