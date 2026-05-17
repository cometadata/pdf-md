from pdf_md import config as config_mod
from pdf_md.config import ExtractConfig


def test_defaults():
    cfg = ExtractConfig()
    assert cfg.format == "markdown"
    assert cfg.workers == 0
    assert cfg.use_ocr is True
    assert cfg.flush_every == 10


def test_with_overrides_ignores_none():
    cfg = ExtractConfig()
    cfg2 = cfg.with_overrides(format=None, workers=8)
    assert cfg2.format == "markdown"   # None ignored
    assert cfg2.workers == 8
    assert cfg is not cfg2             # frozen → new instance


def test_with_overrides_rejects_unknown_field():
    cfg = ExtractConfig()
    try:
        cfg.with_overrides(nonsense=1)
    except TypeError:
        return
    raise AssertionError("expected TypeError for unknown field")


def test_effective_workers_explicit_wins():
    assert ExtractConfig(workers=3).effective_workers() == 3


def test_effective_workers_zero_uses_default(monkeypatch):
    monkeypatch.setattr(config_mod, "_default_workers", lambda: 7)
    assert ExtractConfig(workers=0).effective_workers() == 7


def test_cpu_quota_cgroup_v2(tmp_path):
    # HF Jobs cpu-upgrade: 800000us quota / 100000us period == 8 CPUs
    p = tmp_path / "cpu.max"
    p.write_text("800000 100000\n")
    assert config_mod._cpu_quota_from_cgroup(v2_path=str(p)) == 8


def test_cpu_quota_cgroup_v2_rounds_up(tmp_path):
    p = tmp_path / "cpu.max"
    p.write_text("150000 100000\n")  # 1.5 CPUs -> 2
    assert config_mod._cpu_quota_from_cgroup(v2_path=str(p)) == 2


def test_cpu_quota_cgroup_v2_unlimited_falls_through(tmp_path):
    p = tmp_path / "cpu.max"
    p.write_text("max 100000\n")
    assert config_mod._cpu_quota_from_cgroup(
        v2_path=str(p),
        v1_quota_path=str(tmp_path / "missing_q"),
        v1_period_path=str(tmp_path / "missing_p"),
    ) is None


def test_cpu_quota_cgroup_v1(tmp_path):
    q = tmp_path / "cfs_quota_us"
    q.write_text("400000\n")
    pr = tmp_path / "cfs_period_us"
    pr.write_text("100000\n")
    assert config_mod._cpu_quota_from_cgroup(
        v2_path=str(tmp_path / "missing"),
        v1_quota_path=str(q),
        v1_period_path=str(pr),
    ) == 4


def test_cpu_quota_no_cgroup_returns_none(tmp_path):
    assert config_mod._cpu_quota_from_cgroup(
        v2_path=str(tmp_path / "a"),
        v1_quota_path=str(tmp_path / "b"),
        v1_period_path=str(tmp_path / "c"),
    ) is None


def test_default_workers_prefers_cgroup_quota(monkeypatch):
    monkeypatch.setattr(config_mod, "_cpu_quota_from_cgroup", lambda: 8)
    assert config_mod._default_workers() == 8


def test_effective_download_workers_explicit_wins():
    assert ExtractConfig(workers=8, download_workers=5).effective_download_workers() == 5


def test_effective_download_workers_defaults_to_2x_workers():
    assert ExtractConfig(workers=8).effective_download_workers() == 16


def test_layout_feature_set_default():
    assert ExtractConfig().layout_feature_set == "rf"


def test_layout_feature_set_override():
    cfg = ExtractConfig().with_overrides(layout_feature_set="imf+rf")
    assert cfg.layout_feature_set == "imf+rf"
