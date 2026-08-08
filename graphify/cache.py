# per-file extraction cache - skip unchanged files on re-run
from __future__ import annotations

import atexit
import hashlib
import json
import os
import re
from collections.abc import Mapping
from pathlib import Path
from pathlib import PureWindowsPath

# Output directory name — override with GRAPHIFY_OUT env var for worktrees or
# shared-output setups. Accepts a relative name ("graphify-out-feature") or an
# absolute path ("/shared/graphify-out"). Single source of truth in graphify.paths
# (#1423); re-exported here as _GRAPHIFY_OUT for the existing call sites.
from graphify.paths import GRAPHIFY_OUT as _GRAPHIFY_OUT
from graphify.paths import is_absolute_path
from graphify.persistence import atomic_write_bytes
from graphify.ids import CURRENT_AST_NODE_ID_SCHEMA
from graphify.semantic_schema import (
    PROMPT_SCHEMA_VERSION,
    SemanticSchemaError,
    normalize_semantic_provenance,
    semantic_provenance_digest,
)

# AST cache entries are the output of graphify's own extractor code, so they
# are only valid for the package version and extractor schema that wrote them:
# keying purely on file content means extractor fixes keep serving stale pre-fix
# results, including when the package version is intentionally unchanged. The
# AST cache therefore uses cache/ast/v{version}-{schema}/. The semantic cache is
# deliberately package-version independent — its prompt schema provides the
# compatibility boundary without re-billing extraction on every release.
try:
    from importlib.metadata import version as _pkg_version

    _EXTRACTOR_VERSION = _pkg_version("graphifyy")
except Exception:
    _EXTRACTOR_VERSION = "unknown"

# Package versions are human-controlled in this fork, so a correctness change
# cannot rely on a version bump to invalidate incompatible AST fragments. Keep
# the extractor schema independent and include it in the on-disk namespace.
_AST_CACHE_SCHEMA = CURRENT_AST_NODE_ID_SCHEMA

# Version dirs already swept this process — cleanup runs once per (base, version).
_cleaned_ast_dirs: set[str] = set()


def _cleanup_stale_ast_entries(ast_base: Path, current_dir: Path) -> None:
    """Remove AST cache entries left behind by other graphify versions.

    Sweeps sibling ``v*/`` directories and unversioned ``*.json`` entries
    (the pre-versioning layout) under ``cache/ast/``. Best-effort: failures
    are ignored, stragglers are retried on the next run.
    """
    key = str(current_dir)
    if key in _cleaned_ast_dirs:
        return
    _cleaned_ast_dirs.add(key)
    if not ast_base.is_dir():
        return
    import shutil

    for child in ast_base.iterdir():
        if child == current_dir:
            continue
        try:
            if child.is_dir() and child.name.startswith("v"):
                shutil.rmtree(child, ignore_errors=True)
            elif child.suffix == ".json":
                child.unlink()
        except OSError:
            pass


# A frontmatter delimiter is a whole line of exactly three dashes (optional
# trailing whitespace). Substring checks like startswith("---") /
# find("\n---") also match `----` thematic breaks and `--- text` prose,
# silently dropping everything above them from the hash (#1259).
_FRONTMATTER_DELIM = re.compile(r"^---[ \t]*\r?$", re.MULTILINE)


def _body_content(content: bytes) -> bytes:
    """Strip YAML frontmatter from Markdown content, returning only the body."""
    text = content.decode(errors="replace")
    opener = _FRONTMATTER_DELIM.match(text)
    if opener is None:
        return content
    closer = _FRONTMATTER_DELIM.search(text, opener.end())
    if closer is None:
        return content
    # Slice right after the closing `---` (not after its line) so the output
    # stays byte-identical with the historical implementation for well-formed
    # frontmatter -- existing semantic-cache hashes must not churn.
    return text[closer.start() + 3 :].encode()


# Stat index: maps source root + absolute path to the last observed content
# identity. Size and mtime remain useful diagnostics, but never substitute for
# reading bytes: supported filesystems do not provide a portable stat tuple
# that proves content has not changed.
_stat_index: dict[str, dict] = {}
_stat_index_root: Path | None = None
_stat_index_dirty: bool = False


def _stat_index_file(root: Path) -> Path:
    _out = Path(_GRAPHIFY_OUT)
    base = _out if _out.is_absolute() else Path(root).resolve() / _out
    return base / "cache" / "stat-index.json"


def _ensure_stat_index(root: Path) -> None:
    global _stat_index, _stat_index_root, _stat_index_dirty
    requested_root = Path(root).resolve()
    if _stat_index_root == requested_root:
        return
    first_load = _stat_index_root is None
    if not first_load:
        _flush_stat_index()
    _stat_index_root = requested_root
    p = _stat_index_file(_stat_index_root)
    if p.exists():
        try:
            loaded = json.loads(p.read_text(encoding="utf-8"))
            _stat_index = loaded if isinstance(loaded, dict) else {}
        except (json.JSONDecodeError, OSError):
            _stat_index = {}
    else:
        _stat_index = {}
    if first_load:
        atexit.register(_flush_stat_index)


def _prune_stat_index() -> None:
    global _stat_index_dirty
    for key, entry in list(_stat_index.items()):
        source = entry.get("path") if isinstance(entry, dict) else None
        if not isinstance(source, str):
            _, separator, source = key.partition("\x00")
            if not separator:
                source = ""
        if not source or not Path(source).is_file():
            del _stat_index[key]
            _stat_index_dirty = True


def _flush_stat_index() -> None:
    global _stat_index_dirty, _stat_index_root
    _prune_stat_index()
    if not _stat_index_dirty or _stat_index_root is None:
        return
    p = _stat_index_file(_stat_index_root)
    try:
        atomic_write_bytes(p, json.dumps(_stat_index, separators=(",", ":")).encode())
    except OSError:
        return
    _stat_index_dirty = False


def _normalize_path(path: Path) -> Path:
    """Normalize path for consistent cache keys across Windows path spellings."""
    import sys

    if sys.platform != "win32":
        return path
    s = str(path)
    if s.startswith("\\\\?\\"):
        s = s[4:]  # strip extended-length prefix \\?\
    return Path(os.path.normcase(s))


def _path_has_case_collision(path: Path, root: Path) -> bool:
    """Return whether lowercasing identity would merge two real source paths."""
    try:
        relative = path.resolve().relative_to(root.resolve())
    except ValueError:
        relative = path.resolve()
    current = root.resolve() if not relative.is_absolute() else Path(relative.anchor)
    for part in relative.parts:
        try:
            matches = [
                entry.name
                for entry in current.iterdir()
                if entry.name.casefold() == part.casefold()
            ]
        except OSError:
            return False
        if len(matches) > 1:
            return True
        current = current / part
    return False


def file_hash(
    path: Path,
    root: Path = Path("."),
    *,
    cache_root: Path | None = None,
    record_stat: bool = True,
) -> str:
    """SHA256 of file contents + path relative to root.

    Reads the file bytes on every identity check. The stat index records the
    observation for pruning and diagnostics but cannot prove byte equality.

    Using a relative path (not absolute) makes cache entries portable across
    machines and checkout directories, so shared caches and CI work correctly.
    Falls back to the resolved absolute path if the file is outside root.

    For Markdown files (.md), only the body below the YAML frontmatter is hashed,
    so metadata-only changes (e.g. reviewed, status, tags) do not invalidate the cache.
    """
    global _stat_index_dirty
    p = _normalize_path(Path(path))
    root = _normalize_path(Path(root))
    if not p.is_file():
        raise IsADirectoryError(f"file_hash requires a file, got: {p}")

    _ensure_stat_index(cache_root or root)
    abs_key = f"{root.resolve()}\x00{p.resolve()}"
    st: "os.stat_result | None" = None
    try:
        st = p.stat()
    except OSError:
        pass

    raw = p.read_bytes()
    content = _body_content(raw) if p.suffix.lower() == ".md" else raw
    h = hashlib.sha256()
    h.update(content)
    h.update(b"\x00")
    try:
        rel = p.resolve().relative_to(Path(root).resolve())
        identity_path = rel.as_posix()
    except ValueError:
        identity_path = p.resolve().as_posix()
    if not _path_has_case_collision(p, root):
        identity_path = identity_path.casefold()
    h.update(identity_path.encode())
    digest = h.hexdigest()

    if st is not None and record_stat:
        _stat_index[abs_key] = {
            "path": str(p.resolve()),
            "size": st.st_size,
            "mtime_ns": st.st_mtime_ns,
            "hash": digest,
        }
        _stat_index_dirty = True

    return digest


def _relativize_source_files_in(payload: dict, root: Path) -> None:
    """Mutate ``payload`` to rewrite absolute ``source_file`` fields as
    forward-slash relative paths from ``root``.

    Mirror of :func:`graphify.watch._relativize_source_files` so cached
    extraction fragments persist in portable form (#777). Already-relative
    fields and out-of-root paths pass through unchanged.

    Only ``root`` is resolved — ``source_file`` itself is relativized
    symbolically so in-root symlinks keep their original name rather than
    pointing at the resolved target. Same reasoning as
    :func:`graphify.detect._to_relative_for_storage`.
    """
    try:
        root_resolved = Path(root).resolve()
    except OSError:
        return
    for bucket in ("nodes", "edges", "hyperedges"):
        for item in payload.get(bucket, []):
            if not isinstance(item, dict):
                continue
            source = item.get("source_file")
            if not source:
                continue
            sp = Path(source)
            if not is_absolute_path(str(source)):
                continue
            if not sp.is_absolute():
                continue
            try:
                rel = os.path.relpath(sp, root_resolved)
            except (ValueError, OSError):
                continue  # out-of-root (e.g. Windows cross-drive)
            if rel == ".." or rel.startswith(".." + os.sep) or rel.startswith("../"):
                continue  # escaped root — keep absolute
            item["source_file"] = rel.replace(os.sep, "/")


def _portable_cache_payload(payload: dict, root: Path) -> tuple[dict, bool] | None:
    """Return a portable payload copy, or refuse ambiguous foreign provenance."""
    import copy

    portable = copy.deepcopy(payload)
    changed = False
    root_resolved = Path(root).resolve()
    for bucket in ("nodes", "edges", "hyperedges"):
        for item in portable.get(bucket, []):
            if not isinstance(item, dict):
                continue
            source = item.get("source_file")
            if not isinstance(source, str) or not source:
                continue
            if not is_absolute_path(source):
                normalized_parts = source.replace("\\", "/").split("/")
                if ".." in normalized_parts or PureWindowsPath(source).drive:
                    return None
                continue
            source_path = Path(source)
            if not source_path.is_absolute():
                return None
            try:
                relative = os.path.relpath(source_path, root_resolved)
            except (OSError, ValueError):
                return None
            if relative == ".." or relative.startswith((".." + os.sep, "../")):
                return None
            item["source_file"] = relative.replace(os.sep, "/")
            changed = True
    return portable, changed


def _source_is_within_root(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(Path(root).resolve())
    except (OSError, ValueError):
        return False
    return True


def _absolutize_source_files_in(payload: dict, root: Path) -> None:
    """Inverse of :func:`_relativize_source_files_in`.

    Re-anchor relative ``source_file`` fields against ``root`` so callers
    that load a cached fragment see the same absolute-path shape that a
    fresh in-process extraction would produce. Legacy cache entries with
    absolute ``source_file`` values pass through unchanged.
    """
    try:
        root_resolved = Path(root).resolve()
    except OSError:
        return
    for bucket in ("nodes", "edges", "hyperedges"):
        for item in payload.get(bucket, []):
            if not isinstance(item, dict):
                continue
            source = item.get("source_file")
            if not source:
                continue
            sp = Path(source)
            if is_absolute_path(str(source)):
                continue
            try:
                item["source_file"] = str(root_resolved / sp)
            except (TypeError, OSError):
                continue


def _cache_dir_path(root: Path, kind: str) -> Path:
    _out = Path(_GRAPHIFY_OUT)
    base = _out if _out.is_absolute() else Path(root).resolve() / _out
    directory = base / "cache" / kind
    if kind == "ast":
        directory = directory / f"v{_EXTRACTOR_VERSION}-{_AST_CACHE_SCHEMA}"
    elif kind == "semantic":
        directory = directory / PROMPT_SCHEMA_VERSION
    return directory


def cache_dir(root: Path = Path("."), kind: str = "ast") -> Path:
    """Returns the cache directory for ``kind`` - creates it if needed.

    kind is "ast" or "semantic". Separate subdirectories prevent semantic cache
    entries from overwriting AST cache entries for the same source_file (#582).

    AST entries live in graphify-out/cache/ast/v{version}-{schema}/ — namespaced
    by both package version and extractor schema because package versions are
    human-controlled independently of correctness changes. Semantic entries are
    package-version independent but namespaced by prompt schema under
    graphify-out/cache/semantic/semantic-v*/.
    """
    d = _cache_dir_path(root, kind)
    if kind == "ast":
        _cleanup_stale_ast_entries(d.parent, d)
    d.mkdir(parents=True, exist_ok=True)
    return d


def load_cached(
    path: Path,
    root: Path = Path("."),
    kind: str = "ast",
    *,
    source_root: Path | None = None,
    semantic_provenance: Mapping[str, object] | None = None,
    semantic_requirements: Mapping[str, object] | None = None,
) -> dict | None:
    """Return cached extraction for this file if hash matches, else None.

    Cache key: SHA256 of file contents.
    Cache value: stored as graphify-out/cache/{kind}/{hash}.json (AST entries
    under the per-version subdirectory, see :func:`cache_dir`).

    AST entries written by other graphify versions or extractor schemas —
    including the legacy flat cache/ layout (pre-0.5.3) and the unversioned
    cache/ast/ layout — are deliberately not consulted: they were produced by
    a different extractor and may be stale.
    Returns None if no cache entry or file has changed.
    """
    identity_root = source_root or root
    if not _source_is_within_root(Path(path), identity_root):
        return None
    try:
        h = file_hash(path, identity_root, cache_root=root, record_stat=False)
    except OSError:
        return None
    directory = _cache_dir_path(root, kind)
    if kind == "semantic":
        if semantic_provenance is None:
            candidates = sorted(directory.glob(f"{h}--*.json"))
            if semantic_requirements:
                matching: list[Path] = []
                for candidate in candidates:
                    try:
                        candidate_payload = json.loads(candidate.read_text(encoding="utf-8"))
                        candidate_identity = normalize_semantic_provenance(
                            candidate_payload.get("identity")
                        )
                    except (OSError, json.JSONDecodeError, SemanticSchemaError, AttributeError):
                        continue
                    if all(
                        candidate_identity.get(str(key)) == str(value)
                        for key, value in semantic_requirements.items()
                    ):
                        matching.append(candidate)
                candidates = matching
            if len(candidates) != 1:
                return None
            entry = candidates[0]
        else:
            try:
                identity_digest = semantic_provenance_digest(semantic_provenance)
            except SemanticSchemaError:
                return None
            entry = directory / f"{h}--{identity_digest}.json"
    else:
        entry = directory / f"{h}.json"
    if entry.exists():
        try:
            stored = json.loads(entry.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None
        if kind == "semantic":
            if not isinstance(stored, dict):
                return None
            try:
                identity = normalize_semantic_provenance(stored.get("identity"))
            except SemanticSchemaError:
                return None
            if semantic_provenance is not None:
                try:
                    expected = normalize_semantic_provenance(semantic_provenance)
                except SemanticSchemaError:
                    return None
                if identity != expected:
                    return None
            if entry.stem.rsplit("--", 1)[-1] != semantic_provenance_digest(identity):
                return None
            result = stored.get("fragment")
        else:
            result = stored
        if not isinstance(result, dict) or "error" in result:
            return None
        portable_result = _portable_cache_payload(result, identity_root)
        if portable_result is None:
            return None
        result, migrated = portable_result
        if migrated:
            try:
                atomic_write_bytes(entry, json.dumps(result).encode())
            except OSError:
                return None
        # Re-anchor relative source_file fields so callers see the same
        # absolute-path shape that a fresh in-process extraction produces
        # (#777). Legacy entries with absolute source_file pass through.
        _absolutize_source_files_in(result, identity_root)
        return result
    return None


def save_cached(
    path: Path,
    result: dict,
    root: Path = Path("."),
    kind: str = "ast",
    *,
    source_root: Path | None = None,
    semantic_provenance: Mapping[str, object] | None = None,
) -> bool:
    """Save extraction result for this file.

    Stores as graphify-out/cache/{kind}/{hash}.json where hash = SHA256 of current file contents.
    result should be a dict with 'nodes' and 'edges' lists.

    No-ops if `path` is not a regular file. Subagent-produced semantic fragments
    occasionally carry a directory path in `source_file`; skipping them prevents
    IsADirectoryError from aborting the whole batch.
    """
    p = Path(path)
    identity_root = source_root or root
    if not p.is_file():
        return False
    if not _source_is_within_root(p, identity_root):
        return False
    if "error" in result:
        return False
    # Relativize source_file fields against ``root`` before write so the
    # cache file on disk is portable across machines and checkout
    # directories (#777). The cache key is content-hashed so lookup is
    # already path-independent; this fixes the embedded path leak.
    #
    # Serialize a relativized copy rather than mutating the caller's dict —
    # downstream pipeline steps (notably extract.py's AST prefix remap, which
    # looks up Path(source_file).resolve() in a prefix table) depend on the
    # source_file field's original absolute form. Mutating the input here would
    # silently break those remaps on the first extraction pass.
    portable_result = _portable_cache_payload(result, identity_root)
    if portable_result is None:
        return False
    on_disk, _ = portable_result
    h = file_hash(p, identity_root, cache_root=root)
    target_dir = cache_dir(root, kind)
    if kind == "semantic":
        identity = normalize_semantic_provenance(semantic_provenance)
        identity_digest = semantic_provenance_digest(identity)
        entry = target_dir / f"{h}--{identity_digest}.json"
        payload = {"identity": identity, "fragment": on_disk}
    else:
        entry = target_dir / f"{h}.json"
        payload = on_disk
    atomic_write_bytes(entry, json.dumps(payload, sort_keys=True).encode())
    return True


def cached_files(root: Path = Path(".")) -> set[str]:
    """Return set of file hashes that have a valid cache entry (any kind)."""
    base = Path(root).resolve() / _GRAPHIFY_OUT / "cache"
    hashes: set[str] = set()
    # Legacy flat entries
    if base.is_dir():
        hashes.update(p.stem for p in base.glob("*.json"))
    # Namespaced entries (ast/ recursively, covering per-version subdirs)
    for kind, pattern in (("ast", "**/*.json"), ("semantic", "**/*.json")):
        d = base / kind
        if d.is_dir():
            hashes.update(p.stem.split("--", 1)[0] for p in d.glob(pattern))
    return hashes


def clear_cache(root: Path = Path(".")) -> None:
    """Delete all cache entries (ast/, semantic/, and legacy flat entries)."""
    base = Path(root).resolve() / _GRAPHIFY_OUT / "cache"
    # Legacy flat entries
    if base.is_dir():
        for f in base.glob("*.json"):
            f.unlink()
    # Namespaced entries (ast/ recursively, covering per-version subdirs)
    for kind, pattern in (("ast", "**/*.json"), ("semantic", "**/*.json")):
        d = base / kind
        if d.is_dir():
            for f in d.glob(pattern):
                f.unlink()


def prune_semantic_cache(root: Path, live_hashes: set[str]) -> int:
    """Remove orphaned semantic cache entries, returning the count pruned.

    The semantic cache is keyed by content hash and semantic provenance under a
    prompt-schema namespace. Package releases do not invalidate it; contract
    changes do. Content changes and deletions can still leave orphan entries.

    This sweeps ``cache/semantic/**/*.json`` and deletes entries outside the
    current schema or whose content-hash prefix is not in ``live_hashes``.
    ``*.tmp`` atomic-write temporaries are skipped, and AST cache state is never
    touched.

    Best-effort, mirroring :func:`_cleanup_stale_ast_entries`: each unlink is
    wrapped in ``try/except OSError`` and a failure is ignored. The worst-case
    failure mode is benign — a surviving orphan costs only one re-extraction of
    one doc on a future run, never incorrect output.
    """
    _out = Path(_GRAPHIFY_OUT)
    base = _out if _out.is_absolute() else Path(root).resolve() / _out
    semantic_dir = base / "cache" / "semantic"
    if not semantic_dir.is_dir():
        return 0
    pruned = 0
    current_dir = _cache_dir_path(root, "semantic")
    for entry in semantic_dir.glob("**/*.json"):
        content_hash = entry.stem.split("--", 1)[0]
        if entry.parent == current_dir and content_hash in live_hashes:
            continue
        try:
            entry.unlink()
            pruned += 1
        except OSError:
            pass
    return pruned


def check_semantic_cache(
    files: list[str],
    root: Path = Path("."),
    *,
    source_root: Path | None = None,
    semantic_provenance: Mapping[str, object] | None = None,
    semantic_requirements: Mapping[str, object] | None = None,
) -> tuple[list[dict], list[dict], list[dict], list[str]]:
    """Check semantic extraction cache for a list of absolute file paths.

    Returns (cached_nodes, cached_edges, cached_hyperedges, uncached_files).
    Uncached files need Claude extraction; cached files are merged directly.
    """
    cached_nodes: list[dict] = []
    cached_edges: list[dict] = []
    cached_hyperedges: list[dict] = []
    uncached: list[str] = []

    identity_root = source_root or root
    for fpath in files:
        p = Path(fpath)
        if not is_absolute_path(fpath):
            p = Path(identity_root) / p
        result = load_cached(
            p,
            root,
            kind="semantic",
            source_root=identity_root,
            semantic_provenance=semantic_provenance,
            semantic_requirements=semantic_requirements,
        )
        if result is not None:
            cached_nodes.extend(result.get("nodes", []))
            cached_edges.extend(result.get("edges", []))
            cached_hyperedges.extend(result.get("hyperedges", []))
        else:
            uncached.append(fpath)

    return cached_nodes, cached_edges, cached_hyperedges, uncached


def save_semantic_cache(
    nodes: list[dict],
    edges: list[dict],
    hyperedges: list[dict] | None = None,
    root: Path = Path("."),
    *,
    source_root: Path | None = None,
    semantic_provenance: Mapping[str, object] | None = None,
) -> int:
    """Save semantic extraction results to cache, keyed by source_file.

    Groups nodes and edges by source_file, then saves one cache entry per file
    under the current cache/semantic schema namespace (separate from AST entries
    in cache/ast/) to prevent hash-key collisions (#582).
    Returns the number of files cached.
    """
    from collections import defaultdict

    by_file: dict[str, dict] = defaultdict(lambda: {"nodes": [], "edges": [], "hyperedges": []})
    for n in nodes:
        src = n.get("source_file", "")
        if src:
            by_file[src]["nodes"].append(n)
    for e in edges:
        src = e.get("source_file", "")
        if src:
            by_file[src]["edges"].append(e)
    for h in hyperedges or []:
        src = h.get("source_file", "")
        if src:
            by_file[src]["hyperedges"].append(h)

    identity_root = source_root or root
    saved = 0
    for fpath, result in by_file.items():
        p = Path(fpath)
        if not is_absolute_path(fpath):
            p = Path(identity_root) / p
        if p.is_file():
            if save_cached(
                p,
                result,
                root,
                kind="semantic",
                source_root=identity_root,
                semantic_provenance=semantic_provenance,
            ):
                saved += 1
    return saved
