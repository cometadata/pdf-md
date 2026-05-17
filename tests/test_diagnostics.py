import pytest

from pdf_md import diagnostics


@pytest.fixture(autouse=True)
def _clean_diag(monkeypatch):
    """Each test starts with the env var unset and a clear process accumulator."""
    monkeypatch.delenv("PDF_MD_DIAG", raising=False)
    diagnostics.accumulator()._data.clear()
    yield
    diagnostics.accumulator()._data.clear()


# --- enabled() env gating ---------------------------------------------------

def test_enabled_false_by_default():
    assert diagnostics.enabled() is False


@pytest.mark.parametrize("val", ["1", "true", "TRUE", "yes", "Yes"])
def test_enabled_truthy_values(monkeypatch, val):
    monkeypatch.setenv("PDF_MD_DIAG", val)
    assert diagnostics.enabled() is True


@pytest.mark.parametrize("val", ["0", "false", "no", "", "  "])
def test_enabled_falsy_values(monkeypatch, val):
    monkeypatch.setenv("PDF_MD_DIAG", val)
    assert diagnostics.enabled() is False


# --- Accumulator ------------------------------------------------------------

def test_accumulator_tallies_count_and_total():
    acc = diagnostics.Accumulator()
    acc.record("a", 1.0)
    acc.record("a", 2.0)
    acc.record("b", 0.5)
    snap = acc.snapshot()
    assert snap["a"] == (2, 3.0)
    assert snap["b"] == (1, 0.5)


def test_accumulator_snapshot_is_a_copy():
    acc = diagnostics.Accumulator()
    acc.record("a", 1.0)
    snap = acc.snapshot()
    acc.record("a", 1.0)
    assert snap["a"] == (1, 1.0)  # earlier snapshot unaffected


def test_accumulator_report_empty():
    assert diagnostics.Accumulator().report("T") == "T: (no samples)"


def test_accumulator_report_sorts_by_total_desc():
    acc = diagnostics.Accumulator()
    acc.record("cheap", 0.1)
    acc.record("expensive", 5.0)
    lines = acc.report("title").splitlines()
    assert lines[0] == "title:"
    assert "expensive" in lines[1]
    assert "cheap" in lines[2]


# --- timed() ----------------------------------------------------------------

def test_timed_records_when_enabled(monkeypatch):
    monkeypatch.setenv("PDF_MD_DIAG", "1")
    with diagnostics.timed("block"):
        pass
    snap = diagnostics.accumulator().snapshot()
    assert snap["block"][0] == 1
    assert snap["block"][1] >= 0.0


def test_timed_is_noop_when_disabled():
    with diagnostics.timed("block"):
        pass
    assert diagnostics.accumulator().snapshot() == {}


# --- _wrap_timed ------------------------------------------------------------

def test_wrap_timed_preserves_return_value_and_args():
    wrapped = diagnostics._wrap_timed("fn", lambda x, y=0: x + y)
    assert wrapped(3, y=4) == 7


def test_wrap_timed_records_each_call():
    wrapped = diagnostics._wrap_timed("fn", lambda: None)
    wrapped()
    wrapped()
    assert diagnostics.accumulator().snapshot()["fn"][0] == 2


def test_wrap_timed_records_even_on_exception():
    def _boom():
        raise ValueError("boom")

    wrapped = diagnostics._wrap_timed("fn", _boom)
    with pytest.raises(ValueError):
        wrapped()
    assert diagnostics.accumulator().snapshot()["fn"][0] == 1


def test_wrap_timed_sets_idempotency_markers():
    orig = lambda: None
    wrapped = diagnostics._wrap_timed("fn", orig)
    assert wrapped._pdf_md_diag_wrapped is True
    assert wrapped._pdf_md_diag_orig is orig


# --- install() --------------------------------------------------------------

def test_install_is_noop_when_disabled():
    # _installed must stay False so a later enabled run can still install.
    diagnostics._installed = False
    diagnostics.install()
    assert diagnostics._installed is False
