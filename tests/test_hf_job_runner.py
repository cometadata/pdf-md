import importlib.util
from pathlib import Path

_RUNNER = Path(__file__).resolve().parents[1] / "pdf_md" / "hf_jobs" / "hf_job_runner.py"
_spec = importlib.util.spec_from_file_location("hf_job_runner", _RUNNER)
_runner = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_runner)


def test_job_config_from_env_defaults(monkeypatch):
    monkeypatch.setenv("INPUT_SOURCE", "org/in")
    for k in ["FORMAT", "WORKERS", "MAX_DOCS", "FLUSH_EVERY", "NO_RESUME",
              "HF_REPO_ID", "PRIVATE", "NO_OCR", "FORCE_OCR", "OCR_LANGUAGE"]:
        monkeypatch.delenv(k, raising=False)
    cfg = _runner._job_config_from_env()
    assert cfg["source"] == "org/in"
    assert cfg["format"] == "markdown"
    assert cfg["workers"] is None
    assert cfg["no_resume"] is False
    assert cfg["use_ocr"] is True
    assert cfg["force_ocr"] is False
    assert cfg["ocr_language"] is None


def test_job_config_ocr_env(monkeypatch):
    monkeypatch.setenv("INPUT_SOURCE", "org/in")
    monkeypatch.setenv("NO_OCR", "true")
    monkeypatch.setenv("FORCE_OCR", "false")
    monkeypatch.setenv("OCR_LANGUAGE", "eng+fra")
    cfg = _runner._job_config_from_env()
    assert cfg["use_ocr"] is False
    assert cfg["force_ocr"] is False
    assert cfg["ocr_language"] == "eng+fra"


def test_job_config_from_env_overrides(monkeypatch):
    monkeypatch.setenv("INPUT_SOURCE", "org/in")
    monkeypatch.setenv("FORMAT", "text")
    monkeypatch.setenv("WORKERS", "16")
    monkeypatch.setenv("MAX_DOCS", "100")
    monkeypatch.setenv("NO_RESUME", "true")
    monkeypatch.setenv("PRIVATE", "1")
    cfg = _runner._job_config_from_env()
    assert cfg["format"] == "text"
    assert cfg["workers"] == 16
    assert cfg["max_docs"] == 100
    assert cfg["no_resume"] is True
    assert cfg["private"] is True


def test_job_config_missing_source(monkeypatch):
    monkeypatch.delenv("INPUT_SOURCE", raising=False)
    try:
        _runner._job_config_from_env()
    except RuntimeError:
        return
    raise AssertionError("expected RuntimeError when INPUT_SOURCE unset")


def test_job_config_layout_feature_set(monkeypatch):
    monkeypatch.setenv("INPUT_SOURCE", "org/in")
    monkeypatch.delenv("LAYOUT_FEATURE_SET", raising=False)
    assert _runner._job_config_from_env()["layout_feature_set"] == "rf"
    monkeypatch.setenv("LAYOUT_FEATURE_SET", "imf+rf")
    assert _runner._job_config_from_env()["layout_feature_set"] == "imf+rf"
