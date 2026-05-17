from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class PageResult:
    page_index: int
    content: str


@dataclass
class ConversionResult:
    doc_id: str
    source: str
    pages: List[PageResult] = field(default_factory=list)
    error: Optional[str] = None
