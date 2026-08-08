"""Tests for graphify/cache.py."""

import json
import os
import pytest
from graphify.cache import (
    file_hash,
    cache_dir,
    load_cached,
    save_cached,
    cached_files,
    clear_cache,
    _body_content,
)
from graphify.cache import check_semantic_cache, save_semantic_cache


@pytest.fixture
def tmp_file(tmp_path):
    f = tmp_path / "sample.txt"
    f.write_text("hello world", encoding="utf-8")
    return f


@pytest.fixture
def cache_root(tmp_path):
    return tmp_path


def test_file_hash_consistent(tmp_file):
    """Same file gives same hash on repeated calls."""
    h1 = file_hash(tmp_file)
    h2 = file_hash(tmp_file)
    assert h1 == h2
    assert isinstance(h1, str)
    assert len(h1) == 64  # SHA256 hex digest length


def test_file_hash_changes(tmp_path):
    """Different file contents give different hashes."""
    f1 = tmp_path / "a.txt"
    f2 = tmp_path / "b.txt"
    f1.write_text("content one", encoding="utf-8")
    f2.write_text("content two", encoding="utf-8")
    assert file_hash(f1) != file_hash(f2)


def test_file_hash_detects_same_size_same_mtime_replacement(tmp_path):
    source = tmp_path / "module.py"
    source.write_text("value = 1\n", encoding="utf-8")
    original_stat = source.stat()
    original_hash = file_hash(source, tmp_path, cache_root=tmp_path)

    source.write_text("value = 2\n", encoding="utf-8")
    os.utime(source, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))

    assert source.stat().st_size == original_stat.st_size
    assert source.stat().st_mtime_ns == original_stat.st_mtime_ns
    assert file_hash(source, tmp_path, cache_root=tmp_path) != original_hash


def test_file_hash_preserves_case_on_case_sensitive_filesystems(tmp_path):
    upper = tmp_path / "Case.py"
    lower = tmp_path / "case.py"
    upper.write_text("same bytes\n", encoding="utf-8")
    lower.write_text("same bytes\n", encoding="utf-8")

    if upper.samefile(lower):
        assert file_hash(upper, tmp_path) == file_hash(lower, tmp_path)
    else:
        assert file_hash(upper, tmp_path) != file_hash(lower, tmp_path)


def test_file_hash_casefolds_unambiguous_mixed_case_identity(tmp_path):
    import hashlib

    source = tmp_path / "MixedCase.py"
    content = b"same bytes\n"
    source.write_bytes(content)
    expected = hashlib.sha256(content + b"\x00" + b"mixedcase.py").hexdigest()

    assert file_hash(source, tmp_path) == expected


def test_cache_roundtrip(tmp_file, cache_root):
    """Save then load returns the same result dict."""
    result = {"nodes": [{"id": "n1", "label": "Node1"}], "edges": []}
    save_cached(tmp_file, result, root=cache_root)
    loaded = load_cached(tmp_file, root=cache_root)
    assert loaded == result


def test_cache_miss_is_read_only(tmp_file, tmp_path):
    storage_root = tmp_path / "external-output"

    assert load_cached(tmp_file, root=storage_root, source_root=tmp_path) is None

    assert not (storage_root / "graphify-out").exists()


def test_cache_storage_root_is_independent_from_source_root(tmp_path):
    source_root = tmp_path / "source"
    source_root.mkdir()
    source = source_root / "module.py"
    source.write_text("def run():\n    return 1\n", encoding="utf-8")
    storage_root = tmp_path / "external-output"
    result = {
        "nodes": [{"id": "module_run", "source_file": str(source)}],
        "edges": [],
    }

    save_cached(source, result, root=storage_root, source_root=source_root)
    loaded = load_cached(source, root=storage_root, source_root=source_root)

    assert loaded == result
    entries = list((storage_root / "graphify-out" / "cache" / "ast").rglob("*.json"))
    assert len(entries) == 1
    on_disk = json.loads(entries[0].read_text(encoding="utf-8"))
    assert on_disk["nodes"][0]["source_file"] == "module.py"
    assert not (source_root / "graphify-out").exists()


def test_file_hash_places_stat_index_under_cache_root(tmp_path):
    source_root = tmp_path / "source"
    source_root.mkdir()
    source = source_root / "module.py"
    source.write_text("value = 1\n", encoding="utf-8")
    storage_root = tmp_path / "external-output"

    digest = file_hash(source, source_root, cache_root=storage_root)

    assert len(digest) == 64
    from graphify.cache import _flush_stat_index

    _flush_stat_index()
    assert (storage_root / "graphify-out" / "cache" / "stat-index.json").exists()
    assert not (source_root / "graphify-out").exists()


def test_file_hash_switches_stat_index_storage_roots(tmp_path):
    from graphify.cache import _flush_stat_index

    for name in ("one", "two"):
        source_root = tmp_path / f"source-{name}"
        source_root.mkdir()
        source = source_root / "module.py"
        source.write_text(f"value = {name!r}\n", encoding="utf-8")
        storage_root = tmp_path / f"output-{name}"
        file_hash(source, source_root, cache_root=storage_root)

    _flush_stat_index()
    for name in ("one", "two"):
        assert (tmp_path / f"output-{name}" / "graphify-out" / "cache" / "stat-index.json").exists()


def test_file_hash_stat_fastpath_respects_source_root(tmp_path):
    source_root = tmp_path / "source"
    nested_root = source_root / "nested"
    nested_root.mkdir(parents=True)
    source = nested_root / "module.py"
    source.write_text("value = 1\n", encoding="utf-8")
    storage_root = tmp_path / "external-output"

    broad_hash = file_hash(source, source_root, cache_root=storage_root)
    nested_hash = file_hash(source, nested_root, cache_root=storage_root)

    assert broad_hash != nested_hash


def test_semantic_cache_separates_storage_and_source_roots(tmp_path):
    source_root = tmp_path / "source"
    source_root.mkdir()
    source = source_root / "notes.md"
    source.write_text("# Notes\n\nStable body.\n", encoding="utf-8")
    storage_root = tmp_path / "external-output"
    nodes = [{"id": "notes", "source_file": "notes.md"}]

    saved = save_semantic_cache(nodes, [], root=storage_root, source_root=source_root)
    cached_nodes, cached_edges, cached_hyperedges, uncached = check_semantic_cache(
        [str(source)], root=storage_root, source_root=source_root
    )

    assert saved == 1
    assert cached_nodes == [{"id": "notes", "source_file": str(source)}]
    assert cached_edges == []
    assert cached_hyperedges == []
    assert uncached == []
    entries = list(cache_dir(storage_root, "semantic").glob("*.json"))
    assert len(entries) == 1
    on_disk = json.loads(entries[0].read_text(encoding="utf-8"))
    assert on_disk["fragment"]["nodes"][0]["source_file"] == "notes.md"
    assert on_disk["identity"]["prompt_schema_version"]
    assert not (source_root / "graphify-out").exists()


def test_semantic_cache_identity_separates_backend_model_and_endpoint(tmp_path):
    source = tmp_path / "notes.md"
    source.write_text("# Notes\n", encoding="utf-8")
    result = {"nodes": [{"id": "notes", "source_file": str(source)}], "edges": []}
    first = {
        "backend": "openai",
        "model": "model-a",
        "endpoint_digest": "a" * 64,
    }
    second = {
        "backend": "openai",
        "model": "model-b",
        "endpoint_digest": "b" * 64,
    }

    assert save_cached(
        source,
        result,
        root=tmp_path,
        kind="semantic",
        semantic_provenance=first,
    )
    assert save_cached(
        source,
        result,
        root=tmp_path,
        kind="semantic",
        semantic_provenance=second,
    )
    assert load_cached(source, root=tmp_path, kind="semantic") is None
    assert (
        load_cached(
            source,
            root=tmp_path,
            kind="semantic",
            semantic_provenance=first,
        )
        == result
    )


def test_semantic_cache_identity_separates_mode_and_chunk_contract(tmp_path):
    from graphify.semantic_schema import semantic_provenance

    source = tmp_path / "notes.md"
    source.write_text("# Notes\n", encoding="utf-8")
    result = {"nodes": [{"id": "notes", "source_file": str(source)}], "edges": []}
    standard = semantic_provenance(
        backend="openai",
        model="model-a",
        endpoint="https://api.example/v1",
    )
    deep = semantic_provenance(
        backend="openai",
        model="model-a",
        endpoint="https://api.example/v1",
        deep_mode=True,
    )
    smaller_budget = semantic_provenance(
        backend="openai",
        model="model-a",
        endpoint="https://api.example/v1",
        token_budget=10_000,
    )

    assert save_cached(
        source,
        result,
        root=tmp_path,
        kind="semantic",
        semantic_provenance=standard,
    )
    assert (
        load_cached(
            source,
            root=tmp_path,
            kind="semantic",
            semantic_provenance=deep,
        )
        is None
    )
    assert (
        load_cached(
            source,
            root=tmp_path,
            kind="semantic",
            semantic_provenance=smaller_budget,
        )
        is None
    )
    assert (
        load_cached(
            source,
            root=tmp_path,
            kind="semantic",
            semantic_requirements={"model": "model-a", "extraction_mode": "standard"},
        )
        == result
    )
    assert (
        load_cached(
            source,
            root=tmp_path,
            kind="semantic",
            semantic_requirements={"model": "model-b", "extraction_mode": "standard"},
        )
        is None
    )


def test_semantic_cache_rejects_previous_prompt_schema(tmp_path):
    from graphify.cache import file_hash

    source = tmp_path / "notes.md"
    source.write_text("# Notes\n", encoding="utf-8")
    semantic_dir = cache_dir(tmp_path, "semantic")
    digest = file_hash(source, tmp_path)
    stale = semantic_dir / f"{digest}--{'0' * 64}.json"
    stale.write_text(
        json.dumps(
            {
                "identity": {
                    "prompt_schema_version": "semantic-v0",
                    "backend": "openai",
                    "model": "model-a",
                    "endpoint_digest": "a" * 64,
                },
                "fragment": {"nodes": [], "edges": [], "hyperedges": []},
            }
        ),
        encoding="utf-8",
    )

    assert load_cached(source, root=tmp_path, kind="semantic") is None


def test_semantic_cache_rejects_and_prunes_legacy_provenance_less_entry(tmp_path):
    from graphify.cache import file_hash, prune_semantic_cache

    source = tmp_path / "notes.md"
    source.write_text("# Notes\n", encoding="utf-8")
    digest = file_hash(source, tmp_path)
    legacy_dir = tmp_path / "graphify-out" / "cache" / "semantic"
    legacy_dir.mkdir(parents=True)
    legacy = legacy_dir / f"{digest}.json"
    legacy.write_text(
        json.dumps({"nodes": [{"id": "unsafe-legacy"}], "edges": [], "hyperedges": []}),
        encoding="utf-8",
    )

    assert load_cached(source, root=tmp_path, kind="semantic") is None
    assert prune_semantic_cache(tmp_path, {digest}) == 1
    assert not legacy.exists()


def test_cache_miss_on_change(tmp_file, cache_root):
    """After file content changes, load_cached returns None."""
    result = {"nodes": [], "edges": [{"source": "a", "target": "b"}]}
    save_cached(tmp_file, result, root=cache_root)
    # Modify the file
    tmp_file.write_text("completely different content", encoding="utf-8")
    assert load_cached(tmp_file, root=cache_root) is None


def test_cached_files(tmp_path, cache_root):
    """cached_files returns the set of cached hashes."""
    f1 = tmp_path / "file1.py"
    f2 = tmp_path / "file2.py"
    f1.write_text("alpha", encoding="utf-8")
    f2.write_text("beta", encoding="utf-8")

    save_cached(f1, {"nodes": [], "edges": []}, root=cache_root)
    save_cached(f2, {"nodes": [], "edges": []}, root=cache_root)

    hashes = cached_files(cache_root)
    assert file_hash(f1, cache_root) in hashes
    assert file_hash(f2, cache_root) in hashes


def test_clear_cache(tmp_file, cache_root):
    """clear_cache removes all .json files from graphify-out/cache/ (all subdirs)."""
    save_cached(tmp_file, {"nodes": [], "edges": []}, root=cache_root)
    # Since v0.5.3 entries go into cache/ast/, not the flat cache/ dir
    cache_base = cache_root / "graphify-out" / "cache"
    assert len(list(cache_base.rglob("*.json"))) > 0
    clear_cache(cache_root)
    assert len(list(cache_base.rglob("*.json"))) == 0


def test_md_frontmatter_only_change_same_hash(tmp_path):
    """Changing only frontmatter fields in a .md file does not change the hash."""
    f = tmp_path / "doc.md"
    f.write_text("---\nreviewed: 2026-01-01\n---\n\n# Title\n\nBody text.", encoding="utf-8")
    h1 = file_hash(f)
    f.write_text("---\nreviewed: 2026-04-09\n---\n\n# Title\n\nBody text.", encoding="utf-8")
    h2 = file_hash(f)
    assert h1 == h2


def test_md_body_change_different_hash(tmp_path):
    """Changing the body of a .md file produces a different hash."""
    f = tmp_path / "doc.md"
    f.write_text("---\nreviewed: 2026-01-01\n---\n\n# Title\n\nOriginal body.", encoding="utf-8")
    h1 = file_hash(f)
    f.write_text("---\nreviewed: 2026-01-01\n---\n\n# Title\n\nChanged body.", encoding="utf-8")
    h2 = file_hash(f)
    assert h1 != h2


def test_md_no_frontmatter_hashed_normally(tmp_path):
    """A .md file with no frontmatter is hashed by its full content."""
    f = tmp_path / "doc.md"
    f.write_text("# Just a heading\n\nNo frontmatter here.", encoding="utf-8")
    h1 = file_hash(f)
    f.write_text("# Just a heading\n\nDifferent content.", encoding="utf-8")
    h2 = file_hash(f)
    assert h1 != h2


def test_non_md_file_hashed_fully(tmp_path):
    """Non-.md files are still hashed by their full content."""
    f = tmp_path / "script.py"
    f.write_text("# comment\nx = 1", encoding="utf-8")
    h1 = file_hash(f)
    f.write_text("# changed comment\nx = 1", encoding="utf-8")
    h2 = file_hash(f)
    assert h1 != h2


def test_body_content_strips_frontmatter():
    """_body_content correctly strips YAML frontmatter."""
    content = b"---\ntitle: Test\n---\n\nActual body."
    assert _body_content(content) == b"\n\nActual body."


def test_body_content_no_frontmatter():
    """_body_content returns content unchanged when no frontmatter present."""
    content = b"No frontmatter here."
    assert _body_content(content) == content


# --- #1259: frontmatter delimiters must be whole `---` lines -----------------


def test_body_content_hr_start_is_not_frontmatter():
    """A document opening with a ``----`` thematic break has no frontmatter;
    a later ``---`` hr must not be mistaken for a close delimiter."""
    content = b"----\nIntro paragraph that must be hashed.\n\n---\nbody"
    assert _body_content(content) == content


def test_body_content_dash_title_start_is_not_frontmatter():
    """``--- title`` on the first line is prose, not an open delimiter."""
    content = b"--- title\nIntro that must be hashed.\n\n---\nbody"
    assert _body_content(content) == content


def test_body_content_dash_text_line_is_not_close_delimiter():
    """``--- text`` and ``----`` lines inside opened frontmatter are not the
    close; without a proper close the content passes through unchanged."""
    content = b"---\ntitle: Test\nbody starts here\n--- not a delimiter\n----\nreal content"
    assert _body_content(content) == content


def test_body_content_later_proper_close_skips_dash_text_lines():
    """A ``--- text`` line is skipped; the next whole ``---`` line closes."""
    content = b"---\ntitle: Test\nnote: --- inline\n---\nreal body"
    assert _body_content(content) == b"\nreal body"


def test_body_content_well_formed_output_byte_identical():
    """For well-formed frontmatter the stripped body must stay byte-identical
    to the historical substring implementation, so existing semantic-cache
    hashes do not churn (re-extraction is billed LLM work)."""
    cases = [
        # (input, output of the historical text.find("\n---")+4 algorithm)
        (b"---\ntitle: Test\n---\n\nActual body.", b"\n\nActual body."),
        (b"---\nreviewed: 2026-01-01\n---\n\n# Title\n\nBody text.", b"\n\n# Title\n\nBody text."),
        # close delimiter with trailing whitespace keeps it in the body
        (b"---\ntitle: Test\n---  \nbody", b"  \nbody"),
        # CRLF line endings
        (b"---\r\ntitle: Test\r\n---\r\nbody", b"\r\nbody"),
        # empty frontmatter block
        (b"---\n---\nbody", b"\nbody"),
        # close as the very last line, no trailing newline
        (b"---\ntitle: Test\n---", b""),
    ]
    for content, expected in cases:
        assert _body_content(content) == expected, content


def test_md_edit_above_hr_changes_hash(tmp_path):
    """Editing content above a mid-document ``----`` break must change the
    hash -- previously that region was silently excluded from hashing."""
    f = tmp_path / "doc.md"
    f.write_text("----\nIntro paragraph.\n\n---\nbody", encoding="utf-8")
    h1 = file_hash(f)
    f.write_text("----\nEdited intro paragraph.\n\n---\nbody", encoding="utf-8")
    h2 = file_hash(f)
    assert h1 != h2


# --- #777: portable cache source_file fields --------------------------------
# ``save_cached`` relativizes ``source_file`` entries inside the cache file
# so a committed ``graphify-out/cache/`` is portable across machines and
# CI runners. ``load_cached`` re-absolutizes them so consumers (extract,
# merge into graph.json) see the same shape that fresh extraction emits.


def test_save_cached_relativizes_source_file(tmp_path):
    """The on-disk cache JSON contains forward-slash relative source_file
    entries — no absolute prefix from the saving machine leaks in."""
    import json
    from graphify.cache import save_cached, file_hash, cache_dir

    (tmp_path / "src").mkdir()
    src = tmp_path / "src" / "foo.py"
    src.write_text("def x(): pass\n", encoding="utf-8")
    abs_src = str(src.resolve())
    result = {
        "nodes": [{"id": "n1", "label": "foo", "source_file": abs_src}],
        "edges": [{"source": "n1", "target": "n1", "source_file": abs_src}],
    }
    save_cached(src, result, root=tmp_path, kind="ast")

    h = file_hash(src, tmp_path)
    entry = cache_dir(tmp_path, "ast") / f"{h}.json"
    on_disk = json.loads(entry.read_text(encoding="utf-8"))
    node_sources = {n["source_file"] for n in on_disk["nodes"]}
    edge_sources = {e["source_file"] for e in on_disk["edges"]}
    assert node_sources == {"src/foo.py"}, (
        f"cache nodes must store relative source_file; got {node_sources}"
    )
    assert edge_sources == {"src/foo.py"}


def test_load_cached_absolutizes_source_file(tmp_path):
    """``load_cached`` returns the same absolute-path shape that a fresh
    extraction produces, so consumers don't need to special-case cache
    hits vs. fresh extraction."""
    from graphify.cache import save_cached, load_cached

    (tmp_path / "src").mkdir()
    src = tmp_path / "src" / "foo.py"
    src.write_text("def x(): pass\n", encoding="utf-8")
    abs_src = str(src.resolve())
    save_cached(
        src,
        {
            "nodes": [{"id": "n1", "source_file": abs_src}],
            "edges": [{"source": "n1", "target": "n1", "source_file": abs_src}],
        },
        root=tmp_path,
        kind="ast",
    )

    loaded = load_cached(src, root=tmp_path, kind="ast")
    assert loaded is not None
    assert loaded["nodes"][0]["source_file"] == abs_src
    assert loaded["edges"][0]["source_file"] == abs_src


@pytest.mark.parametrize("foreign", [r"C:\repo\src\app.py", r"\\server\share\app.py"])
def test_cache_path_normalizers_preserve_foreign_absolute_paths(tmp_path, foreign):
    from graphify.cache import _absolutize_source_files_in, _relativize_source_files_in

    payload = {"nodes": [{"source_file": foreign}], "edges": [], "hyperedges": []}
    _relativize_source_files_in(payload, tmp_path)
    _absolutize_source_files_in(payload, tmp_path)

    assert payload["nodes"][0]["source_file"] == foreign


def test_load_cached_migrates_legacy_in_root_absolute_source_file(tmp_path):
    """Cache entries written by an older graphify (with absolute source_file
    inside) must still load correctly: the absolutize step is a no-op for
    already-absolute values."""
    import json
    from graphify.cache import load_cached, file_hash, cache_dir

    (tmp_path / "src").mkdir()
    src = tmp_path / "src" / "foo.py"
    src.write_text("pass\n", encoding="utf-8")
    abs_src = str(src.resolve())

    # Hand-write a legacy-format cache entry (absolute source_file).
    h = file_hash(src, tmp_path)
    entry = cache_dir(tmp_path, "ast") / f"{h}.json"
    entry.write_text(
        json.dumps(
            {
                "nodes": [{"id": "n1", "source_file": abs_src}],
                "edges": [],
            }
        ),
        encoding="utf-8",
    )

    loaded = load_cached(src, root=tmp_path, kind="ast")
    assert loaded is not None
    assert loaded["nodes"][0]["source_file"] == abs_src
    migrated = json.loads(entry.read_text(encoding="utf-8"))
    assert migrated["nodes"][0]["source_file"] == "src/foo.py"


@pytest.mark.parametrize(
    "foreign",
    [r"C:\foreign\src\foo.py", r"\\host\share\foo.py", "/foreign/src/foo.py"],
)
def test_load_cached_refuses_foreign_absolute_provenance(tmp_path, foreign):
    src = tmp_path / "foo.py"
    src.write_text("pass\n", encoding="utf-8")
    h = file_hash(src, tmp_path)
    entry = cache_dir(tmp_path, "ast") / f"{h}.json"
    entry.write_text(
        json.dumps({"nodes": [{"id": "n1", "source_file": foreign}], "edges": []}),
        encoding="utf-8",
    )

    assert load_cached(src, root=tmp_path, kind="ast") is None


def test_save_cached_refuses_error_and_out_of_root_provenance(tmp_path):
    src = tmp_path / "foo.py"
    src.write_text("pass\n", encoding="utf-8")

    assert not save_cached(src, {"nodes": [], "edges": [], "error": "failed"}, root=tmp_path)
    assert not save_cached(
        src,
        {"nodes": [{"id": "n1", "source_file": "/foreign/foo.py"}], "edges": []},
        root=tmp_path,
    )
    assert list((tmp_path / "graphify-out" / "cache" / "ast").rglob("*.json")) == []


def test_load_cached_refuses_legacy_error_entry(tmp_path):
    src = tmp_path / "foo.py"
    src.write_text("pass\n", encoding="utf-8")
    h = file_hash(src, tmp_path)
    entry = cache_dir(tmp_path, "ast") / f"{h}.json"
    entry.write_text(
        json.dumps({"nodes": [], "edges": [], "error": "old failure"}),
        encoding="utf-8",
    )

    assert load_cached(src, root=tmp_path, kind="ast") is None


@pytest.mark.parametrize("source_file", ["../outside.py", r"C:relative\outside.py"])
def test_load_cached_refuses_relative_provenance_escape(tmp_path, source_file):
    src = tmp_path / "foo.py"
    src.write_text("pass\n", encoding="utf-8")
    h = file_hash(src, tmp_path)
    entry = cache_dir(tmp_path, "ast") / f"{h}.json"
    entry.write_text(
        json.dumps({"nodes": [{"id": "n1", "source_file": source_file}], "edges": []}),
        encoding="utf-8",
    )

    assert load_cached(src, root=tmp_path, kind="ast") is None


def test_cache_refuses_source_outside_declared_source_root(tmp_path):
    source_root = tmp_path / "source"
    source_root.mkdir()
    outside = tmp_path / "outside.py"
    outside.write_text("pass\n", encoding="utf-8")

    assert not save_cached(outside, {"nodes": [], "edges": []}, root=source_root)
    assert load_cached(outside, root=source_root) is None


def test_save_cached_replace_failure_preserves_existing_entry(tmp_path, monkeypatch):
    import graphify.cache as cache_mod
    import graphify.persistence as persistence_mod

    src = tmp_path / "foo.py"
    src.write_text("pass\n", encoding="utf-8")
    save_cached(src, {"nodes": [{"id": "old"}], "edges": []}, root=tmp_path)
    entry = cache_dir(tmp_path, "ast") / f"{file_hash(src, tmp_path)}.json"
    before = entry.read_bytes()

    def fail_replace(source, target):
        raise PermissionError("simulated replace failure")

    monkeypatch.setattr(persistence_mod.os, "replace", fail_replace)
    monkeypatch.setattr(cache_mod.os, "replace", fail_replace)
    with pytest.raises(PermissionError, match="simulated replace failure"):
        save_cached(src, {"nodes": [{"id": "new"}], "edges": []}, root=tmp_path)

    assert entry.read_bytes() == before


def test_stat_index_prunes_deleted_sources(tmp_path, monkeypatch):
    import graphify.cache as cache_mod

    source = tmp_path / "gone.py"
    source.write_text("pass\n", encoding="utf-8")
    monkeypatch.setattr(cache_mod, "_stat_index", {})
    monkeypatch.setattr(cache_mod, "_stat_index_root", None)
    monkeypatch.setattr(cache_mod, "_stat_index_dirty", False)
    file_hash(source, tmp_path, cache_root=tmp_path)
    source.unlink()
    cache_mod._flush_stat_index()

    index_path = tmp_path / "graphify-out" / "cache" / "stat-index.json"
    assert json.loads(index_path.read_text(encoding="utf-8")) == {}


def test_cache_portable_across_roots(tmp_path):
    """End-to-end portability: a cache entry written at one root can be
    consumed at a different absolute root because the file is content-hashed
    AND its embedded source_file is stored relative."""
    import shutil
    from graphify.cache import save_cached, load_cached

    repo_a = tmp_path / "repo_a"
    repo_a.mkdir()
    (repo_a / "src").mkdir()
    src_a = repo_a / "src" / "foo.py"
    src_a.write_text("def x(): pass\n", encoding="utf-8")
    save_cached(
        src_a,
        {
            "nodes": [{"id": "n1", "source_file": str(src_a.resolve())}],
            "edges": [],
        },
        root=repo_a,
        kind="ast",
    )

    # Copy corpus + cache to a second location with a different absolute prefix.
    repo_b = tmp_path / "repo_b"
    shutil.copytree(repo_a, repo_b)

    src_b = repo_b / "src" / "foo.py"
    loaded = load_cached(src_b, root=repo_b, kind="ast")
    assert loaded is not None, (
        "cache must port across absolute prefixes (content hash + relative source_file)"
    )
    # Source path re-anchored to the new root, not the old one.
    assert loaded["nodes"][0]["source_file"] == str(src_b.resolve())
    assert str(repo_a) not in loaded["nodes"][0]["source_file"]


# --- AST cache versioning ----------------------------------------------------
# AST cache entries are the output of graphify's own extractor code, so they
# are valid only for both the package version and the extractor schema that
# wrote them. The schema component matters when a correctness fix ships without
# a human-authorized package-version change. The semantic cache has its own
# prompt-schema namespace instead.


def test_ast_cache_invalidated_on_version_bump(tmp_path, monkeypatch):
    """An AST entry written by version X must not be served after upgrading
    to version Y — the file is unchanged but the extractor is not."""
    import graphify.cache as cache_mod

    f = tmp_path / "mod.py"
    f.write_text("def f(): pass\n", encoding="utf-8")

    monkeypatch.setattr(cache_mod, "_EXTRACTOR_VERSION", "0.8.0", raising=False)
    save_cached(f, {"nodes": [{"id": "n1"}], "edges": []}, root=tmp_path, kind="ast")
    assert load_cached(f, root=tmp_path, kind="ast") is not None

    monkeypatch.setattr(cache_mod, "_EXTRACTOR_VERSION", "0.8.1", raising=False)
    assert load_cached(f, root=tmp_path, kind="ast") is None, (
        "AST cache entry from a previous graphify version must not be served"
    )


def test_ast_cache_invalidated_on_schema_change_without_version_bump(tmp_path, monkeypatch):
    """An extractor-schema change must invalidate AST entries even when the
    package version is intentionally unchanged."""
    import graphify.cache as cache_mod

    f = tmp_path / "pyproject.toml"
    f.write_text('[project]\nname = "demo"\n', encoding="utf-8")

    monkeypatch.setattr(cache_mod, "_EXTRACTOR_VERSION", "0.9.5", raising=False)
    monkeypatch.setattr(cache_mod, "_AST_CACHE_SCHEMA", "node-id-v0", raising=False)
    save_cached(
        f,
        {"nodes": [{"id": "pkg_demo"}], "edges": []},
        root=tmp_path,
        kind="ast",
    )
    legacy_dir = cache_dir(tmp_path, "ast")
    assert load_cached(f, root=tmp_path, kind="ast") is not None

    monkeypatch.setattr(cache_mod, "_AST_CACHE_SCHEMA", "node-id-v1", raising=False)
    assert load_cached(f, root=tmp_path, kind="ast") is None

    monkeypatch.setattr(cache_mod, "_cleaned_ast_dirs", set(), raising=False)
    current_dir = cache_dir(tmp_path, "ast")
    assert current_dir != legacy_dir
    assert not legacy_dir.exists()


def test_ast_cache_version_bump_cleans_stale_entries(tmp_path, monkeypatch):
    """Upgrading removes AST entries left behind by previous versions so the
    cache directory does not grow one full copy per release."""
    import graphify.cache as cache_mod

    f = tmp_path / "mod.py"
    f.write_text("def f(): pass\n", encoding="utf-8")

    monkeypatch.setattr(cache_mod, "_EXTRACTOR_VERSION", "0.8.0", raising=False)
    save_cached(f, {"nodes": [{"id": "n1"}], "edges": []}, root=tmp_path, kind="ast")
    old_dir = cache_dir(tmp_path, "ast")
    assert any(old_dir.glob("*.json"))

    monkeypatch.setattr(cache_mod, "_EXTRACTOR_VERSION", "0.8.1", raising=False)
    monkeypatch.setattr(cache_mod, "_cleaned_ast_dirs", set(), raising=False)
    cache_dir(tmp_path, "ast")
    assert not old_dir.exists(), "stale AST version directory must be removed on upgrade"


def test_legacy_unversioned_ast_entries_not_served(tmp_path):
    """Entries written by pre-versioning graphify (flat cache/ or unversioned
    cache/ast/) are by definition from an older extractor and must not be
    served — that staleness is exactly what version namespacing fixes."""
    import json
    from graphify.cache import file_hash, _GRAPHIFY_OUT

    f = tmp_path / "mod.py"
    f.write_text("def f(): pass\n", encoding="utf-8")
    h = file_hash(f, tmp_path)
    payload = json.dumps({"nodes": [{"id": "stale"}], "edges": []})

    # Unversioned cache/ast/{hash}.json (pre-versioning layout)
    unversioned = tmp_path / _GRAPHIFY_OUT / "cache" / "ast"
    unversioned.mkdir(parents=True)
    (unversioned / f"{h}.json").write_text(payload, encoding="utf-8")
    # Legacy flat cache/{hash}.json (pre-0.5.3 layout)
    (unversioned.parent / f"{h}.json").write_text(payload, encoding="utf-8")

    assert load_cached(f, root=tmp_path, kind="ast") is None


def test_semantic_cache_survives_version_bump(tmp_path, monkeypatch):
    """The semantic cache is deliberately not versioned: entries are produced
    by the LLM from file contents, and re-extraction costs real money."""
    import graphify.cache as cache_mod

    f = tmp_path / "doc.md"
    f.write_text("# Title\n\nBody.\n", encoding="utf-8")

    monkeypatch.setattr(cache_mod, "_EXTRACTOR_VERSION", "0.8.0", raising=False)
    save_cached(f, {"nodes": [{"id": "n1"}], "edges": []}, root=tmp_path, kind="semantic")
    semantic_dir = cache_dir(tmp_path, "semantic")

    monkeypatch.setattr(cache_mod, "_EXTRACTOR_VERSION", "0.8.1", raising=False)
    monkeypatch.setattr(cache_mod, "_cleaned_ast_dirs", set(), raising=False)
    cache_dir(tmp_path, "ast")  # triggers stale-AST cleanup
    assert load_cached(f, root=tmp_path, kind="semantic") is not None
    assert any(semantic_dir.glob("*.json")), (
        "semantic entries must survive both the version bump and AST cleanup"
    )


def test_save_cached_in_root_symlink_keeps_symlink_name(tmp_path, path_alias):
    """``source_file`` for an in-root symlink must be stored under the
    symlink's own name, not the resolved target. Lower-impact than the
    manifest case (cache lookup is content-hashed, not key-matched), but
    keeps the on-disk shape consistent with what callers passed in."""
    import json
    from graphify.cache import save_cached, file_hash, cache_dir

    (tmp_path / "sub").mkdir()
    target = tmp_path / "sub" / "target.py"
    target.write_text("pass\n", encoding="utf-8")
    alias = path_alias(tmp_path / "alias.py", target)

    abs_alias = str(alias)  # caller's view — the symlink path, unresolved
    save_cached(
        alias,
        {
            "nodes": [{"id": "n1", "source_file": abs_alias}],
            "edges": [],
        },
        root=tmp_path,
        kind="ast",
    )

    h = file_hash(alias, tmp_path)
    entry = cache_dir(tmp_path, "ast") / f"{h}.json"
    on_disk = json.loads(entry.read_text(encoding="utf-8"))
    expected_alias = alias.relative_to(tmp_path).as_posix()
    assert on_disk["nodes"][0]["source_file"] == expected_alias, (
        f"cache must store symlink name, not resolved target; got "
        f"{on_disk['nodes'][0]['source_file']!r}"
    )


def test_semantic_prune_removes_orphan_entries(tmp_path):
    """Changing a file's content leaves the old content-hash entry orphaned;
    pruning against the new live hash removes the stale entry and keeps the
    current one."""
    from graphify.cache import prune_semantic_cache

    f = tmp_path / "doc.md"
    f.write_text("# A\n\nContent A.\n", encoding="utf-8")
    h_a = file_hash(f, tmp_path)
    save_cached(f, {"nodes": [{"id": "a"}], "edges": []}, root=tmp_path, kind="semantic")

    f.write_text("# B\n\nContent B.\n", encoding="utf-8")
    h_b = file_hash(f, tmp_path)
    save_cached(f, {"nodes": [{"id": "b"}], "edges": []}, root=tmp_path, kind="semantic")

    semantic_dir = cache_dir(tmp_path, "semantic")
    entry_a = next(semantic_dir.glob(f"{h_a}--*.json"))
    entry_b = next(semantic_dir.glob(f"{h_b}--*.json"))
    assert entry_a.exists()
    assert entry_b.exists()

    pruned = prune_semantic_cache(tmp_path, {h_b})
    assert pruned == 1
    assert not entry_a.exists()
    assert entry_b.exists()


def test_semantic_prune_keeps_live_unchanged_entries(tmp_path):
    """Pruning against the FULL live set must keep every live entry — guards
    the trap of pruning against an incremental changed-subset, which would
    delete all unchanged docs' valid entries."""
    from graphify.cache import prune_semantic_cache

    live_hashes = set()
    for i in range(5):
        f = tmp_path / f"doc{i}.md"
        f.write_text(f"# Doc {i}\n\nBody {i}.\n", encoding="utf-8")
        save_cached(f, {"nodes": [{"id": str(i)}], "edges": []}, root=tmp_path, kind="semantic")
        live_hashes.add(file_hash(f, tmp_path))

    semantic_dir = cache_dir(tmp_path, "semantic")
    assert len(list(semantic_dir.glob("*.json"))) == 5

    pruned = prune_semantic_cache(tmp_path, live_hashes)
    assert pruned == 0
    assert len(list(semantic_dir.glob("*.json"))) == 5


def test_semantic_prune_handles_deleted_file(tmp_path):
    """An entry for a file that no longer exists (dropped from the live set) is
    pruned."""
    from graphify.cache import prune_semantic_cache

    f = tmp_path / "gone.md"
    f.write_text("# Gone\n\nWill be deleted.\n", encoding="utf-8")
    h = file_hash(f, tmp_path)
    save_cached(f, {"nodes": [{"id": "g"}], "edges": []}, root=tmp_path, kind="semantic")
    semantic_dir = cache_dir(tmp_path, "semantic")
    entry = next(semantic_dir.glob(f"{h}--*.json"))
    assert entry.exists()

    f.unlink()
    # Live set is empty: the file is gone, so its entry must be pruned.
    pruned = prune_semantic_cache(tmp_path, set())
    assert pruned == 1
    assert not entry.exists()


def test_semantic_prune_ignores_ast_and_tmp(tmp_path):
    """Prune touches only cache/semantic/*.json: AST entries and atomic-write
    *.tmp temporaries are left untouched."""
    from graphify.cache import prune_semantic_cache

    f = tmp_path / "doc.md"
    f.write_text("# Doc\n\nBody.\n", encoding="utf-8")
    # AST entry (different subtree) must survive.
    save_cached(f, {"nodes": [{"id": "ast"}], "edges": []}, root=tmp_path, kind="ast")
    ast_dir = cache_dir(tmp_path, "ast")
    assert len(list(ast_dir.glob("*.json"))) == 1

    # A semantic orphan .json (to be pruned) plus a .tmp temporary (to survive).
    semantic_dir = cache_dir(tmp_path, "semantic")
    (semantic_dir / "deadbeef.json").write_text('{"nodes": [], "edges": []}', encoding="utf-8")
    tmp_entry = semantic_dir / "deadbeef.tmp"
    tmp_entry.write_text("partial", encoding="utf-8")

    pruned = prune_semantic_cache(tmp_path, set())
    assert pruned == 1
    assert not (semantic_dir / "deadbeef.json").exists()
    assert tmp_entry.exists(), "*.tmp temporaries must not be swept"
    assert len(list(ast_dir.glob("*.json"))) == 1, "AST entries must not be touched"
