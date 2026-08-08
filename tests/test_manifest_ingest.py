from __future__ import annotations

from pathlib import Path

from graphify.build import build_from_json
from graphify.detect import FileType, classify_file
from graphify.extract import extract
from graphify.manifest_ingest import (
    _pkg_id,
    extract_package_manifest,
    is_package_manifest_path,
)


def _write(p: Path, text: str) -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    return p


# ── routing: manifests are deterministic (CODE), not LLM documents ───────────


def test_manifests_classify_as_code_not_document(tmp_path):
    for name in ("apm.yml", "pyproject.toml", "go.mod", "pom.xml"):
        p = _write(tmp_path / name, "x")
        assert is_package_manifest_path(p)
        assert classify_file(p) is FileType.CODE, name
    # a generic yaml stays a document
    assert classify_file(_write(tmp_path / "config.yaml", "a: 1")) is FileType.DOCUMENT


# ── per-format parsing ───────────────────────────────────────────────────────


def _pkg_nodes(result):
    return [n for n in result["nodes"] if n.get("type") == "package"]


def test_apm_parses_name_and_deps(tmp_path):
    p = _write(
        tmp_path / "apm.yml", "name: my-pkg\nversion: 1.2.3\ndependencies:\n  - dep-a\n  - dep-b\n"
    )
    r = extract_package_manifest(p)
    pkg = _pkg_nodes(r)[0]
    assert pkg["label"] == "my-pkg" and pkg["version"] == "1.2.3"
    deps = {e["target"] for e in r["edges"] if e["relation"] == "depends_on"}
    assert {_pkg_id("apm", "dep-a"), _pkg_id("apm", "dep-b")} <= deps


def test_pyproject_parses_pep508_deps(tmp_path):
    p = _write(
        tmp_path / "pyproject.toml",
        '[project]\nname = "cool-lib"\nversion = "0.1"\n'
        'dependencies = ["requests>=2.0", "rich[jupyter]==13.0", "tomli; python_version<\'3.11\'"]\n',
    )
    r = extract_package_manifest(p)
    assert _pkg_nodes(r)[0]["label"] == "cool-lib"
    deps = {e["target"] for e in r["edges"]}
    assert {
        _pkg_id("python", "requests"),
        _pkg_id("python", "rich"),
        _pkg_id("python", "tomli"),
    } <= deps  # versions/extras/markers stripped


def test_gomod_parses_module_and_requires(tmp_path):
    p = _write(
        tmp_path / "go.mod",
        "module example.com/me/app\n\ngo 1.22\n\nrequire (\n"
        "\tgithub.com/x/y v1.2.3\n\tgithub.com/a/b v0.4.0\n)\n",
    )
    r = extract_package_manifest(p)
    assert _pkg_nodes(r)[0]["label"] == "example.com/me/app"
    deps = {e["target"] for e in r["edges"]}
    assert _pkg_id("go", "github.com/x/y") in deps
    assert _pkg_id("go", "github.com/a/b") in deps


def test_pom_parses_artifact_and_deps(tmp_path):
    p = _write(
        tmp_path / "pom.xml",
        '<project xmlns="http://maven.apache.org/POM/4.0.0">\n'
        "  <groupId>com.acme</groupId>\n  <artifactId>widget</artifactId>\n  <version>2.0</version>\n"
        "  <dependencies>\n    <dependency><groupId>org.lib</groupId><artifactId>core</artifactId></dependency>\n"
        "  </dependencies>\n</project>\n",
    )
    r = extract_package_manifest(p)
    assert _pkg_nodes(r)[0]["label"] == "com.acme:widget"
    assert any(e["target"] == _pkg_id("maven", "org.lib:core") for e in r["edges"])


def test_same_package_name_is_namespaced_by_ecosystem(tmp_path):
    pyproject = _write(
        tmp_path / "python/pyproject.toml",
        '[project]\nname = "shared"\ndependencies = ["peer"]\n',
    )
    gomod = _write(
        tmp_path / "go/go.mod",
        "module shared\n\nrequire peer v1.0.0\n",
    )

    python_result = extract_package_manifest(pyproject)
    go_result = extract_package_manifest(gomod)

    assert _pkg_nodes(python_result)[0]["id"] == _pkg_id("python", "shared")
    assert _pkg_nodes(go_result)[0]["id"] == _pkg_id("go", "shared")
    assert _pkg_id("python", "shared") != _pkg_id("go", "shared")
    assert {edge["target"] for edge in python_result["edges"]} == {_pkg_id("python", "peer")}
    assert {edge["target"] for edge in go_result["edges"]} == {_pkg_id("go", "peer")}


# ── #1377: a package referenced by N manifests is ONE node ───────────────────


def test_apm_dependency_collapses_to_single_canonical_node(tmp_path):
    base = tmp_path / "packages"
    _write(base / "core/apm.yml", "name: coding-standards-core\nversion: 1.0.4\n")
    _write(
        base / "csharp/apm.yml",
        "name: coding-standards-csharp\ndependencies:\n  - coding-standards-core\n",
    )
    _write(
        base / "python/apm.yml",
        'name: coding-standards-python\ndependencies:\n  coding-standards-core: ">=1.0"\n',
    )
    files = sorted(base.rglob("apm.yml"))
    result = extract(files, cache_root=tmp_path)

    core = [
        n
        for n in result["nodes"]
        if n.get("type") == "package" and n["label"] == "coding-standards-core"
    ]
    assert len(core) == 1, "core package must be a single canonical node"
    assert core[0]["id"] == _pkg_id("apm", "coding-standards-core")
    assert core[0]["source_file"]

    g = build_from_json(result)
    core_ids = [n for n, d in g.nodes(data=True) if d.get("label") == "coding-standards-core"]
    dep_edges = [(u, v) for u, v, d in g.edges(data=True) if d.get("relation") == "depends_on"]
    assert len(core_ids) == 1
    assert len(dep_edges) == 2  # both dependents point at the one core node


def test_external_dependency_edge_pruned_not_orphaned(tmp_path):
    # A dep whose manifest isn't in the corpus: the edge dangles and build prunes it.
    p = _write(tmp_path / "apm.yml", "name: leaf\ndependencies:\n  - some-external-pkg\n")
    result = extract([p], cache_root=tmp_path)
    g = build_from_json(result)
    assert _pkg_id("apm", "some-external-pkg") not in set(g.nodes())
    assert [n for n, d in g.nodes(data=True) if d.get("label") == "leaf"]


def test_malformed_manifest_does_not_crash(tmp_path):
    p = _write(tmp_path / "pom.xml", "<project><not closed")
    r = extract_package_manifest(p)  # parse error -> empty, no exception
    assert r["nodes"] == [] and r["edges"] == []


def test_pom_rejects_entity_expansion(tmp_path):
    p = _write(
        tmp_path / "pom.xml",
        "<!DOCTYPE project [<!ENTITY injected 'owned'>]>"
        "<project><artifactId>&injected;</artifactId></project>",
    )

    result = extract_package_manifest(p)

    assert result["nodes"] == []
    assert result["edges"] == []
    assert "parse error" in result["error"]
