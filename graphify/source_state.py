"""Ground-truth source lifecycle states shared by update/watch integrations."""

from __future__ import annotations

from enum import Enum
from pathlib import Path


class SourceState(str, Enum):
    DETECTED_EXISTING = "detected_existing"
    EXTRACTABLE_AST = "extractable_ast"
    CHANGED_AST = "changed_ast"
    SEMANTIC_REFRESH_PENDING = "semantic_refresh_pending"
    IGNORED = "ignored"
    PROVEN_DELETED = "proven_deleted"


def classify_changed_source(
    path: Path,
    *,
    detected_existing: set[Path],
    extractable_ast: set[Path],
    ignored: bool,
) -> SourceState:
    """Classify a changed path without treating non-AST content as deleted."""
    candidate = Path(path).resolve()
    if candidate.exists():
        if ignored:
            return SourceState.IGNORED
        if candidate in extractable_ast:
            return SourceState.CHANGED_AST
        if candidate in detected_existing:
            return SourceState.SEMANTIC_REFRESH_PENDING
        return SourceState.DETECTED_EXISTING
    return SourceState.PROVEN_DELETED


def destructive_source_state(state: SourceState) -> bool:
    """Only ignored and filesystem-proven deletion authorize source eviction."""
    return state in {SourceState.IGNORED, SourceState.PROVEN_DELETED}
