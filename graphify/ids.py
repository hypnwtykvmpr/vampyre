"""Single source of truth for node-ID normalization.

Three independent producers must agree on node IDs or the graph splits a single
entity into disconnected ghost nodes:

1. The AST extractor (``extract._make_id``) — deterministic, per-language.
2. The semantic subagents (LLM) — follow the node-ID spec in the skill prompt.
3. The graph builder (``build._normalize_id``) — reconciles edge endpoints when
   the LLM emits IDs with slightly different punctuation or casing than the AST.

Historically the normalization recipe was copy-pasted into ``extract._make_id``
and ``build._normalize_id`` and kept in sync only by mirrored docstrings, which
is exactly how the recurring ID-drift bug class crept in (#811 Unicode collapse,
#550 same-filename collisions, #1033 AST-vs-LLM file-node mismatch, #1104). This
module exists so the recipe lives in one place and the two callers can no longer
diverge.

The recipe: NFKC-normalize (so composed/decomposed Unicode forms collapse),
casefold, NFKC-normalize again, replace runs of non-word characters with a
single underscore (``re.UNICODE`` so CJK/Cyrillic/Arabic/accented-Latin letters
survive instead of collapsing to a per-file node), collapse repeated
underscores, and strip leading/trailing underscores.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata

CURRENT_AST_NODE_ID_SCHEMA = "unicode-boundary-sha1-v1"
AST_NODE_ID_SCHEMA_PROFILE_KEY = "ast_node_id_schema"

__all__ = [
    "AST_NODE_ID_SCHEMA_PROFILE_KEY",
    "CURRENT_AST_NODE_ID_SCHEMA",
    "disambiguate_path_scoped_ids",
    "normalize_id",
    "make_id",
]


def normalize_id(s: str) -> str:
    r"""Normalize a single ID string to its canonical form.

    Idempotent: ``normalize_id(normalize_id(s)) == normalize_id(s)``.
    """
    s = unicodedata.normalize("NFKC", s).casefold()
    s = unicodedata.normalize("NFKC", s)
    s = re.sub(r"[^\w]+", "_", s, flags=re.UNICODE)
    s = re.sub(r"_+", "_", s)
    return s.strip("_")


def make_id(*parts: str) -> str:
    """Build a canonical node ID from one or more name parts.

    A single part is normalized directly. Multiple parts append a boundary-aware
    digest so ``("mod", "a_b")`` differs from ``("mod_a", "b")`` while the
    flattened prefix remains available to source-root remapping. Empty canonical
    parts are rejected.
    """
    normalized = [normalize_id(part.strip("_.")) for part in parts if part]
    if not normalized or any(not part for part in normalized):
        raise ValueError("node ID cannot have an empty canonical part")
    if len(normalized) == 1:
        return normalized[0]
    # Preserve the historical first-part prefix used by source-root remapping,
    # while binding the flattened spelling to the original part boundaries.
    # The unit separator cannot occur in normalized parts, so the digest input
    # distinguishes ("mod", "a_b") from ("mod_a", "b").
    flattened = "_".join(normalized)
    # The first part is the replaceable source/file namespace. Keeping it out
    # of the suffix makes an absolute-prefix ID remap byte-identical to building
    # the ID from the canonical repository-relative prefix in the first place.
    signature = "\x1f".join(normalized[1:])
    # Keep the full digest. A short display hash becomes a material collision
    # risk on large multi-repository graphs, where an ID collision would merge
    # unrelated entities rather than merely affect presentation.
    digest = hashlib.sha1(signature.encode("utf-8"), usedforsecurity=False).hexdigest()
    return f"{flattened}_{digest}"


def disambiguate_path_scoped_ids(base_id: str, source_keys: set[str]) -> dict[str, str]:
    """Qualify one contested ID by source path using the AST collision contract."""
    naive = {source_key: make_id(source_key, base_id) for source_key in source_keys if source_key}
    counts: dict[str, int] = {}
    for node_id in naive.values():
        counts[node_id] = counts.get(node_id, 0) + 1
    result: dict[str, str] = {}
    for source_key, node_id in naive.items():
        if counts[node_id] == 1:
            result[source_key] = node_id
            continue
        salt = hashlib.sha1(source_key.encode("utf-8"), usedforsecurity=False).hexdigest()[:6]
        result[source_key] = make_id(source_key, base_id, salt)
    return result
