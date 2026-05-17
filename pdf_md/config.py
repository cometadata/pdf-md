from __future__ import annotations

import math
import os
from dataclasses import dataclass, fields, replace
from typing import Optional


def _cpu_quota_from_cgroup(
    v2_path: str = "/sys/fs/cgroup/cpu.max",
    v1_quota_path: str = "/sys/fs/cgroup/cpu/cpu.cfs_quota_us",
    v1_period_path: str = "/sys/fs/cgroup/cpu/cpu.cfs_period_us",
) -> Optional[int]:
    # Containers (HF Jobs, Docker, k8s) report the *host* core count via
    # os.cpu_count(), but the process is throttled to a smaller cgroup quota.
    # Sizing the worker pool to the host count grossly over-subscribes the
    # container: on an 8-CPU HF Job, os.cpu_count() returns 64, so the pool
    # spawns 64 worker processes that exhaust the container's memory and
    # thread budget (native thread spawns then fail with EAGAIN).
    # cgroup v2: "<quota> <period>" or "max <period>"
    try:
        with open(v2_path) as f:
            quota_s, period_s = f.read().split()
        if quota_s != "max":
            return max(1, math.ceil(int(quota_s) / int(period_s)))
    except (OSError, ValueError):
        pass
    # cgroup v1
    try:
        with open(v1_quota_path) as f:
            quota = int(f.read())
        with open(v1_period_path) as f:
            period = int(f.read())
        if quota > 0 and period > 0:
            return max(1, math.ceil(quota / period))
    except (OSError, ValueError):
        pass
    return None


def _default_workers() -> int:
    quota = _cpu_quota_from_cgroup()
    if quota is not None:
        return quota
    try:
        return len(os.sched_getaffinity(0)) or 1
    except AttributeError:  # not available on macOS/Windows
        return os.cpu_count() or 1


@dataclass(frozen=True)
class ExtractConfig:
    format: str = "markdown"          # markdown | text | json | chunks
    workers: int = 0                  # 0 -> os.cpu_count()
    download_workers: int = 0         # 0 -> 2 * effective_workers()
    flush_every: int = 10             # batches per HF shard flush
    worker_batch_size: int = 8        # docs accumulated per yielded batch
    layout_feature_set: str = "rf"  # rf (default, text-only) | imf+rf (pymupdf's default, image CNN)
    use_ocr: bool = True
    force_ocr: bool = False
    ocr_language: str = "eng"
    ocr_dpi: int = 300
    dpi: int = 150
    write_images: bool = False
    image_path: Optional[str] = None
    image_format: str = "png"
    doc_timeout: int = 120
    retry_isolated: bool = True

    def with_overrides(self, **kwargs) -> "ExtractConfig":
        valid = {f.name for f in fields(self)}
        unknown = set(kwargs) - valid
        if unknown:
            raise TypeError(f"Unknown ExtractConfig fields: {sorted(unknown)}")
        applied = {k: v for k, v in kwargs.items() if v is not None}
        return replace(self, **applied)

    def effective_workers(self) -> int:
        return self.workers if self.workers > 0 else _default_workers()

    def effective_download_workers(self) -> int:
        # Downloads are I/O-bound; one download thread per conversion worker
        # leaves workers starved, so the default look-ahead is 2x the
        # conversion-worker count.
        if self.download_workers > 0:
            return self.download_workers
        return 2 * self.effective_workers()
