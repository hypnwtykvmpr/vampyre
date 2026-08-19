"""Tests for the tools/skillgen generator and the claude lean-core split.

skillgen renders graphify's committed skill artifacts from human-edited
fragments. These tests lock in the current anti-drift guard (``--check``),
render idempotency, and the lean-core invariant: the
core runs a default extraction with zero reference reads, on-demand content
lives only in the references, and no reference duplicates core content.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path


# tests/ -> repo root is one parent up; put it on the path so tools.skillgen
# imports regardless of pytest's import mode.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.skillgen import gen  # noqa: E402
from graphify.installation import VAMPYRE_UV_SOURCE  # noqa: E402
from graphify.semantic_schema import EDGE_RELATIONS, render_semantic_schema  # noqa: E402


def test_check_passes():
    """The committed artifacts and the expected/ snapshot match a fresh render.

    This is the CI / pre-commit drift guard. A failure here means someone
    hand-edited a generated file or forgot to re-run the generator.
    """
    platforms = gen.load_platforms()
    artifacts = gen.render_all(platforms, only="claude")
    problems = gen.check(artifacts)
    assert problems == [], "\n".join(problems)


def test_render_is_idempotent():
    """Rendering twice yields byte-identical output with no dynamic metadata."""
    platforms = gen.load_platforms()
    first = gen.render_all(platforms, only="claude")
    second = gen.render_all(platforms, only="claude")
    assert [(a.path, a.content) for a in first] == [(a.path, a.content) for a in second]


def test_render_output_is_lf_only():
    """Generated artifacts use LF newlines and end in exactly one newline."""
    platforms = gen.load_platforms()
    for art in gen.render_all(platforms, only="claude"):
        assert "\r" not in art.content, art.path
        assert art.content.endswith("\n"), art.path
        assert not art.content.endswith("\n\n"), art.path


def test_release_source_is_canonical_and_version_is_not_duplicated():
    """Release versions appear only as the canonical immutable install source."""
    from graphify.__main__ import __version__

    platforms = gen.load_platforms()
    saw_release_source = False
    for art in gen.render_all(platforms, only="claude"):
        install_sources = re.findall(
            r"git\+https://github\.com/hypnwtykvmpr/vampyre\.git@v[0-9][0-9A-Za-z.+-]*",
            art.content,
        )
        assert set(install_sources) <= {VAMPYRE_UV_SOURCE}, (
            f"{art.path} carries a non-canonical Vampyre install source: {install_sources}"
        )
        saw_release_source = saw_release_source or bool(install_sources)
        without_release_source = art.content.replace(VAMPYRE_UV_SOURCE, "")
        assert __version__ not in without_release_source, (
            f"{art.path} carries the package version outside the canonical install source"
        )
    assert saw_release_source, "generated guidance lost the immutable release install source"


def _claude_artifacts():
    platforms = gen.load_platforms()
    arts = gen.render_all(platforms, only="claude")
    core = next(a for a in arts if a.path == "graphify/skill.md")
    refs = {a.path.rsplit("/", 1)[-1]: a.content for a in arts if a.path != "graphify/skill.md"}
    return core.content, refs


def test_lean_core_has_no_reference_only_content():
    """The core must not inline the execution detail of an on-demand reference.

    The ``## Usage`` flag table in the core deliberately lists every command,
    including the on-demand ones (it is the --help payload), so the markers
    below are execution-detail lines that never appear in that table.
    """
    core, _ = _claude_artifacts()
    # The full embedded subagent prompt lives only in extraction-spec.md.
    assert '"file_type":"code|document|paper|image|rationale|concept"' not in core
    # The incremental-update merge machinery lives only in update.md.
    assert "from graphify.build import build_merge" not in core
    assert "graphify cluster-only ." not in core
    # The vocab-expansion query flow lives only in query.md.
    assert "Constrained query expansion" not in core
    assert "save-result --question" not in core
    # The export commands live only in exports.md.
    assert "graphify export wiki" not in core
    assert "graphify export neo4j" not in core
    # The add / watch / hook flows live only in their references.
    assert "from graphify.ingest import ingest" not in core
    assert "graphify hook install" not in core
    assert "python3 -m graphify.watch" not in core


def test_lean_core_runs_default_pipeline_with_zero_references():
    """The default code-corpus run must be fully described inside the core."""
    core, _ = _claude_artifacts()
    # The whole default pipeline (detect -> AST -> build -> label -> HTML ->
    # report) must be present in the core so a plain run reads no reference.
    for needed in (
        "### Step 1 - Ensure graphify is installed",
        "### Step 2 - Detect files",
        "### Step 3 - Extract entities and relationships",
        "#### Part A - Structural extraction for code files",
        "#### Part C - Merge AST + semantic into final extraction",
        "### Step 4 - Build graph, cluster, analyze, generate outputs",
        "### Step 5 - Label communities",
        "### Step 6 - Generate Obsidian vault (opt-in) + HTML",
        "### Step 9 - Save manifest, update cost tracker, clean up, and report",
        "## Honesty Rules",
        "graphify export html",
    ):
        assert needed in core, f"lean core is missing default-pipeline content: {needed!r}"


def test_extraction_states_no_api_key_required_for_every_host():
    """Regression for #1461: every skill body that describes Step 3 extraction must
    state up front that no API key is required, tell the agent never to prompt for or
    block on one, and give a terminal-only (non-subagent) fallback.

    Hermes (and the other AGENTS.md hosts) run the CLI directly and can't dispatch
    subagents; the old text framed the no-key path only as 'dispatch subagents as
    written', so those agents looped for minutes insisting on a missing API key.
    """
    platforms = gen.load_platforms()
    arts = gen.render_all(platforms)
    bodies = [a for a in arts if "### Step 3 - Extract entities and relationships" in a.content]
    assert bodies, "no rendered skill body contains the Step 3 extraction section"
    for a in bodies:
        assert "graphify needs no API key" in a.content, a.path
        assert "Never ask the user for one, and never block on one." in a.content, a.path
        # the no-key fallback must not be framed *only* around subagent dispatch
        assert "cannot dispatch subagents" in a.content, a.path
        # where a host prints the GEMINI key tip, the clarity must precede it (be
        # hoisted) rather than sit buried after the key check (aider/devin print no
        # tip — they are the model themselves — so the check only applies if present)
        tip = "Tip: set `GEMINI_API_KEY`"
        if tip in a.content:
            assert a.content.index("graphify needs no API key") < a.content.index(tip), (
                f"{a.path}: no-key clarity is not hoisted above the GEMINI tip"
            )


def test_references_contain_no_core_pipeline_content():
    """No reference fragment may duplicate the core build pipeline."""
    _, refs = _claude_artifacts()
    # Distinctive lines from the core build/label steps must not appear in any
    # reference, or the same content would be double-homed.
    core_only_markers = (
        "from graphify.cluster import cluster, score_all",
        "### Step 4 - Build graph, cluster, analyze, generate outputs",
        "### Step 5 - Label communities",
        "## Honesty Rules",
    )
    for name, body in refs.items():
        for marker in core_only_markers:
            assert marker not in body, f"reference {name} leaked core content: {marker!r}"


def test_reference_pointers_in_core_resolve_to_real_fragments():
    """Every references/<name>.md the core points at is actually rendered."""
    import re

    core, refs = _claude_artifacts()
    pointed = set(re.findall(r"references/([\w-]+)\.md", core))
    rendered = {name[: -len(".md")] for name in refs}
    missing = pointed - rendered
    assert not missing, f"core points at references that were not rendered: {missing}"


def test_query_heading_is_homed_in_core_stub_only():
    """The query section heading is the lean-core stub; query.md re-homes the rest."""
    core, refs = _claude_artifacts()
    core_headings = set(gen.headings(core))
    query_headings = set(gen.headings(refs["query.md"]))
    assert "## For /graphify query" in core_headings
    assert "## For /graphify query" not in query_headings
    # The deeper query content moved into the reference.
    assert "## For /graphify path" in query_headings
    assert "## For /graphify explain" in query_headings
    assert "## For /graphify path" not in core_headings


def test_eight_references_render_for_claude():
    """claude renders exactly the eight on-demand fragments from the design."""
    _, refs = _claude_artifacts()
    assert sorted(refs) == [
        "add-watch.md",
        "exports.md",
        "extraction-spec.md",
        "github-and-merge.md",
        "hooks.md",
        "query.md",
        "transcribe.md",
        "update.md",
    ]


def test_headings_helper_ignores_code_fence_comments():
    """The fence-aware heading scanner must skip '#' lines inside code fences."""
    md = (
        "# Real Heading\n"
        "\n"
        "```bash\n"
        "# not a heading, a shell comment\n"
        "echo hi\n"
        "```\n"
        "\n"
        "## Another Real One\n"
    )
    assert gen.headings(md) == ["# Real Heading", "## Another Real One"]


def test_enum_is_full_six_value_superset_in_extraction_spec():
    """Decision A: the file_type enum is the full six-value superset."""
    _, refs = _claude_artifacts()
    spec = refs["extraction-spec.md"]
    assert "`code`, `document`, `paper`, `image`, `rationale`, `concept`" in spec
    assert '"file_type":"code|document|paper|image|rationale|concept"' in spec


def test_extraction_reference_schema_is_rendered_from_runtime_owner():
    """Public references and runtime extraction share executable schema data."""
    _, refs = _claude_artifacts()
    spec = refs["extraction-spec.md"]
    canonical = render_semantic_schema()

    assert canonical in spec
    assert "@@SEMANTIC_SCHEMA@@" not in spec
    assert "|".join(EDGE_RELATIONS) in spec


def test_every_split_extraction_variant_renders_the_runtime_schema():
    variants = set()
    for platform in gen.load_platforms().values():
        if platform.refs_dst is None:
            continue
        variants.add(platform.extraction)
        artifacts = gen.render_all({platform.key: platform}, only=platform.key)
        spec = next(
            artifact.content
            for artifact in artifacts
            if artifact.path.endswith("/extraction-spec.md")
        )
        assert "@@SEMANTIC_SCHEMA@@" not in spec
        assert render_semantic_schema() in spec
    assert variants == {"compact", "verbose"}


# --- codex + windows (the divergent split hosts) -------------------------------


def _platform_artifacts(key):
    platforms = gen.load_platforms()
    arts = gen.render_all(platforms, only=key)
    skill_dst = platforms[key].skill_dst
    core = next(a for a in arts if a.path == skill_dst)
    refs = {a.path.rsplit("/", 1)[-1]: a.content for a in arts if a.path != skill_dst}
    return core.content, refs


def test_check_passes_for_codex_and_windows():
    """The committed codex/windows artifacts match a fresh render and expected/."""
    platforms = gen.load_platforms()
    for key in ("codex", "windows"):
        artifacts = gen.render_all(platforms, only=key)
        problems = gen.check(artifacts)
        assert problems == [], f"[{key}]\n" + "\n".join(problems)


UNIFIED_DESCRIPTION = (
    "Use for any question about a codebase, its architecture, file relationships, "
    "or project content — especially when graphify-out/ exists, where the question "
    "should be treated as a graphify query first. Turns any input (code, docs, "
    "papers, images, videos) into a persistent knowledge graph with god nodes, "
    "community detection, and query/path/explain tools."
)


def test_descriptions_are_unified():
    """Every platform now carries one unified frontmatter description, byte for byte.

    The two legacy descriptions (Claude's short one and the richer 14-host
    line) were collapsed into a single discovery-tuned line that leads with the
    use-condition. Every split host and both monoliths must carry it verbatim,
    and none of the old wording may survive.
    """
    expected_line = f'description: "{UNIFIED_DESCRIPTION}"'
    platforms = gen.load_platforms()
    for key, p in platforms.items():
        body = gen.render(p)[0].content
        assert expected_line in body, f"[{key}] missing the unified description line"
        # None of the drifted legacy wording may survive on any platform.
        assert "Provides persistent graph with god nodes" not in body, f"[{key}] kept old wording"
        assert "treat the question as a /graphify query." not in body, f"[{key}] kept old wording"
        assert "clustered communities" not in body, f"[{key}] kept old wording"


def test_windows_frontmatter_name_and_shell_and_extra():
    """windows: graphify-windows uses one coherent Git Bash command dialect."""
    core, _ = _platform_artifacts("windows")
    assert core.startswith("---\nname: graphify-windows\n")
    assert "Claude Code on Windows runs skill commands in Git Bash" in core
    assert "```powershell" not in core
    assert "function Find-GraphifyPython" not in core
    assert "cygpath" in core
    assert "## Troubleshooting" in core
    assert "### PowerShell 5.1: Vertical scrolling stops working" in core
    assert "pip install" not in core
    assert "pip uninstall" not in core
    # The troubleshooting section sits before Honesty Rules, single separator.
    assert "\n4. **Skip native Leiden**" in core
    assert core.index("## Troubleshooting") < core.index("## Honesty Rules")


def test_generated_skills_quote_saved_interpreter_paths():
    """Interpreter paths containing spaces must remain one command argument."""
    platforms = gen.load_platforms()
    for platform in platforms.values():
        for artifact in gen.render(platform):
            for line in artifact.content.splitlines():
                stripped = line.lstrip()
                if stripped.startswith("$(cat graphify-out/.graphify_python)"):
                    raise AssertionError(f"{artifact.path}: unquoted interpreter in {line!r}")


def test_codex_dispatch_is_agenttask_and_collects_in_memory():
    """codex: spawn/wait/close_agent dispatch needing multi_agent = true."""
    core, _ = _platform_artifacts("codex")
    assert "spawn_agent" in core
    assert "wait_agent" in core
    assert "close_agent" in core
    assert "multi_agent = true" in core
    assert "Codex collects in memory" in core
    # The B2 dispatch slot itself (Codex heading -> Step B3) must not carry the
    # claude Agent-tool example. The shared Step B3 prose mentions the agent type
    # in a re-run hint, so scope the check to the dispatch block only.
    b2 = core[core.index("**Step B2") : core.index("**Step B3")]
    assert "Concrete example for 3 chunks" not in b2
    assert "Agent tool call 1" not in b2


def test_codex_and_windows_unify_enum_to_six_values():
    """codex (was 4-value) and windows (was 5-value) now carry the superset."""
    for key in ("codex", "windows"):
        _, refs = _platform_artifacts(key)
        spec = refs["extraction-spec.md"]
        assert "`code`, `document`, `paper`, `image`, `rationale`, `concept`" in spec
        assert '"file_type":"code|document|paper|image|rationale|concept"' in spec
        # No legacy 4-value enum survives anywhere in the rendered bundle.
        for body in refs.values():
            assert '"file_type":"code|document|paper|image"' not in body


def test_codex_uses_compact_extraction_windows_uses_verbose():
    """The extraction variant differs: codex compact, windows verbose."""
    _, codex_refs = _platform_artifacts("codex")
    _, windows_refs = _platform_artifacts("windows")
    assert "(compact)" in codex_refs["extraction-spec.md"]
    assert "(compact)" not in windows_refs["extraction-spec.md"]


def test_every_platform_query_has_expansion_and_fallback():
    """#1325: the unified query reference ships BOTH the vocab-expansion step and
    the inline NetworkX fallback to every platform (previously split so no host
    got both — Claude had expansion but no fallback; the rest the reverse)."""
    for key in ("claude", "codex", "windows", "opencode"):
        core, refs = _platform_artifacts(key)
        # Core stub mentions both the vocab-expansion step and the inline fallback.
        assert "expand the question against the graph's own vocabulary" in core
        assert "NetworkX traversal" in core
        # The query reference carries expansion, fallback, and path/explain.
        q = refs["query.md"]
        assert "Constrained query expansion" in q
        assert "If the CLI is unavailable" in q
        assert "## For /graphify path" in q
        assert "## For /graphify explain" in q


def test_schema_singleton_passes_across_all_platforms():
    """The file_type enum is the six-value superset in every rendered artifact."""
    platforms = gen.load_platforms()
    problems = gen.schema_singleton(platforms)
    assert problems == [], "\n".join(problems)


def test_schema_singleton_catches_legacy_enums():
    """The guard's line scanner flags 4- and 5-value pipe enums, not the superset."""
    four = 'file_type":"code|document|paper|image"'
    five = 'file_type":"code|document|paper|image|rationale"'
    superset = '"file_type":"code|document|paper|image|rationale|concept"'
    assert gen.legacy_enum_lines(four) == [four]
    assert gen.legacy_enum_lines(five) == [five]
    # The full six-value superset is never flagged.
    assert gen.legacy_enum_lines(superset) == []
    assert gen.legacy_enum_lines("no enum here") == []


# --- the remaining progressive hosts -------------------------------------------

_PROGRESSIVE_HOSTS = (
    "opencode",
    "kilo",
    "copilot",
    "claw",
    "droid",
    "amp",
    "trae",
    "kiro",
    "pi",
    "vscode",
)


def test_all_progressive_hosts_match_current_snapshots():
    """Every progressive host matches its committed current snapshot."""
    platforms = gen.load_platforms()
    for key in _PROGRESSIVE_HOSTS:
        arts = gen.render_all(platforms, only=key)
        assert gen.check(arts) == [], f"[{key}] check\n" + "\n".join(gen.check(arts))


def test_no_host_has_trigger_in_frontmatter():
    """No split host emits a trigger: field — not part of Agent Skills spec (#1180)."""
    for key in (
        "claude",
        "codex",
        "opencode",
        "kilo",
        "copilot",
        "claw",
        "droid",
        "amp",
        "trae",
        "vscode",
        "kiro",
        "pi",
    ):
        core, _ = _platform_artifacts(key)
        head = core.split("---", 2)[1]
        assert "trigger:" not in head, f"[{key}] unexpectedly has a trigger: line"


def test_kilo_renders_its_rules_tail_section():
    """kilo gets the Kilo-specific rules tail before Honesty Rules."""
    core, _ = _platform_artifacts("kilo")
    assert "## Kilo-specific rules" in core
    assert core.index("## Kilo-specific rules") < core.index("## Honesty Rules")


def test_dispatch_variants_are_host_specific():
    """Each dispatch variant lands in the right host's B2 slot."""
    expect = {
        "opencode": "@mention",
        "droid": "Task(description=",
        "amp": "Task(description=",
        "trae": "Task(description=",
        "vscode": "paste each response back",
    }
    for key, marker in expect.items():
        core, _ = _platform_artifacts(key)
        b2 = core[core.index("**Step B2") : core.index("**Step B3")]
        assert marker.lower() in b2.lower(), f"[{key}] dispatch slot missing {marker!r}"


def test_compact_extraction_hosts_use_the_compact_spec():
    """kiro, pi, claw use the compact extraction body; the rest use verbose."""
    for key in ("kiro", "pi", "claw"):
        _, refs = _platform_artifacts(key)
        assert "(compact)" in refs["extraction-spec.md"], f"[{key}] not compact"
    for key in ("opencode", "kilo", "copilot", "droid", "amp", "trae", "vscode"):
        _, refs = _platform_artifacts(key)
        assert "(compact)" not in refs["extraction-spec.md"], f"[{key}] should be verbose"


def test_every_split_host_renders_eight_references():
    """All twelve split hosts render exactly the eight on-demand references."""
    platforms = gen.load_platforms()
    expected = [
        "add-watch.md",
        "exports.md",
        "extraction-spec.md",
        "github-and-merge.md",
        "hooks.md",
        "query.md",
        "transcribe.md",
        "update.md",
    ]
    for key, p in platforms.items():
        if p.bucket != "split":
            continue
        _, refs = _platform_artifacts(key)
        assert sorted(refs) == expected, f"[{key}] reference set drift: {sorted(refs)}"


# --- the aider + devin monoliths -----------------------------------------------


def test_monoliths_render_inline_single_file_no_references():
    """aider and devin render one inline body, no split and no references dir."""
    platforms = gen.load_platforms()
    for key in ("aider", "devin"):
        assert platforms[key].bucket == "monolith"
        arts = gen.render(platforms[key])
        assert len(arts) == 1, f"[{key}] monolith should render exactly one file"
        assert arts[0].path == f"graphify/skill-{key}.md"
        assert (
            "references/" not in arts[0].content
            or "see `references/" not in arts[0].content.lower()
        )


def test_monoliths_carry_the_1392_runbook_fixes():
    """The four #1392 data-loss/correctness fixes are present in both monoliths.

    The round-trip allows these change-classes; this test asserts they are
    actually applied, so a regression that drops a fix fails here even though the
    round-trip (which only forbids *unsanctioned* drift) would still pass.
    """
    platforms = gen.load_platforms()
    for key in ("aider", "devin"):
        body = gen.render(platforms[key])[0].content

        # Graph-class propagation: no bare build survives, and both graph-class
        # substitutions are present.
        assert "directed=IS_DIRECTED" in body
        assert "multigraph=IS_MULTIGRAPH" in body
        assert "build_from_json(extraction)" not in body
        assert "replace `IS_DIRECTED` and `IS_MULTIGRAPH` everywhere" in body

        # #10 content-only semantic scope: code is no longer flattened in.
        assert "for cat in ('document', 'paper', 'image')" in body
        assert "detect['files'].values()" not in body

        # #12 stale-cache unlink on a miss.
        assert ".graphify_cached.json').unlink(missing_ok=True)" in body

        # #18/#20 zero-node guard before any write, report/analysis gated on
        # to_json's return.
        lines = body.splitlines()
        build_i = next(
            i
            for i, line in enumerate(lines)
            if (
                "G = build_from_json(extraction, directed=IS_DIRECTED, "
                "multigraph=IS_MULTIGRAPH)" in line
            )
        )
        guard_i = next(
            i for i, line in enumerate(lines[build_i:], build_i) if "number_of_nodes() == 0" in line
        )
        report_i = next(
            i
            for i, line in enumerate(lines[build_i:], build_i)
            if "atomic_write_text(report_path, report)" in line
        )
        wrote_i = next(
            i
            for i, line in enumerate(lines[build_i:], build_i)
            if line.strip().startswith("if not to_json(")
        )
        # guard fires right after the build, before the graph/report are written.
        assert build_i < guard_i < wrote_i < report_i, f"[{key}] Step 4 ordering not fixed"
        assert "if not to_json(" in body


def test_generated_skills_use_strict_state_and_preserve_multigraph_updates():
    platforms = gen.load_platforms()
    for key, platform in platforms.items():
        body = "\n".join(artifact.content for artifact in gen.render(platform))
        assert "/graphify <path> --multigraph" in body, key
        assert "resolve_update_context" in body, key
        assert "multigraph=IS_MULTIGRAPH" in body, key
        assert "expected_hashes=detect['_source_snapshot']" in body, key
        assert "write_scan_root_marker" in body, key
        assert "> graphify-out/.graphify_root" not in body, key
        assert "Out-File -FilePath graphify-out\\.graphify_root" not in body, key

    update_fragment = (
        REPO_ROOT / "tools/skillgen/fragments/references/shared/update.md"
    ).read_text(encoding="utf-8")
    assert "save_manifest(" not in update_fragment


def test_every_generated_runbook_commits_then_acknowledges_under_publication_lock():
    for key, platform in gen.load_platforms().items():
        body = "\n".join(artifact.content for artifact in gen.render(platform))
        sections = {
            "build": body.split("### Step 4 - Build graph", maxsplit=1)[1].split(
                "### Step 5 - Label communities", maxsplit=1
            )[0],
            "labels": body.split("### Step 5 - Label communities", maxsplit=1)[1].split(
                "### Step 6 -", maxsplit=1
            )[0],
            "manifest": body.split("### Step 9 - Save manifest", maxsplit=1)[1],
        }
        for section_name, section in sections.items():
            lines = section.splitlines()
            lock_i = next(i for i, line in enumerate(lines) if "with output_state_lock(" in line)
            transaction_i = next(
                i
                for i, line in enumerate(lines[lock_i + 1 :], lock_i + 1)
                if "FileStateTransaction(" in line
            )
            commit_i = next(
                i
                for i, line in enumerate(lines[transaction_i + 1 :], transaction_i + 1)
                if "transaction.commit()" in line
            )
            lock_indent = len(lines[lock_i]) - len(lines[lock_i].lstrip())
            transaction_indent = len(lines[transaction_i]) - len(lines[transaction_i].lstrip())
            commit_indent = len(lines[commit_i]) - len(lines[commit_i].lstrip())
            assert lock_i < transaction_i < commit_i, (key, section_name)
            assert transaction_indent > lock_indent, (key, section_name)
            assert commit_indent > transaction_indent, (key, section_name)

            if section_name == "manifest":
                acknowledge_i = next(
                    i
                    for i, line in enumerate(lines[commit_i + 1 :], commit_i + 1)
                    if "acknowledge_pending_signal(" in line
                )
                acknowledge_indent = len(lines[acknowledge_i]) - len(lines[acknowledge_i].lstrip())
                assert commit_i < acknowledge_i, key
                assert acknowledge_indent == transaction_indent, key


def test_generated_runbooks_pass_root_to_save_manifest():
    """#1417: every save_manifest call in a shipped runbook threads root=.

    Without root=, save_manifest stores absolute path keys, so a clone or move
    breaks --update (every cached file misses and the whole corpus re-extracts).
    The full-build runbooks relativize the manifest to the scan root via
    root='INPUT_PATH'. Incremental references deliberately do not save a
    manifest before canonical graph publication. This guards the actual shipped
    artifacts; --check keeps them in sync with the fragments.
    """
    targets = [
        REPO_ROOT / "graphify" / "skill.md",
        REPO_ROOT / "graphify" / "skill-aider.md",
        REPO_ROOT / "graphify" / "skill-devin.md",
    ]
    targets += sorted((REPO_ROOT / "graphify" / "skills").glob("*/references/update.md"))
    checked = 0
    for path in targets:
        for ln in path.read_text(encoding="utf-8").splitlines():
            if "save_manifest(" in ln and "import" not in ln:
                checked += 1
                assert "root=" in ln, (
                    f"{path.relative_to(REPO_ROOT)}: save_manifest without root= (#1417): {ln.strip()!r}"
                )
    assert checked == 3, f"expected one full-build save per shipped core, found {checked}"


def test_devin_keeps_its_multi_field_frontmatter():
    """devin renders inline, so its 4+-field frontmatter is preserved verbatim."""
    platforms = gen.load_platforms()
    body = gen.render(platforms["devin"])[0].content
    head = body.split("---", 2)[1]
    assert "argument-hint:" in head
    assert "model:" in head
    assert "allowed-tools:" in head


# --- the always-on instruction blocks (D2-a) -----------------------------------


def test_always_on_renders_six_blocks():
    """render_always_on yields exactly the six always-on instruction files."""
    arts = gen.render_always_on()
    paths = sorted(a.path for a in arts)
    assert paths == [
        "graphify/always_on/agents-md.md",
        "graphify/always_on/antigravity-rules.md",
        "graphify/always_on/claude-md.md",
        "graphify/always_on/gemini-md.md",
        "graphify/always_on/kiro-steering.md",
        "graphify/always_on/vscode-instructions.md",
    ]


def test_always_on_included_in_full_render_not_per_platform():
    """A full render carries the always-on files; a --platform render does not."""
    platforms = gen.load_platforms()
    full = {a.path for a in gen.render_all(platforms)}
    claude_only = {a.path for a in gen.render_all(platforms, only="claude")}
    assert "graphify/always_on/claude-md.md" in full
    assert "graphify/always_on/claude-md.md" not in claude_only


def test_extracted_constants_equal_the_packaged_always_on_files():
    """The live module constants now equal the packaged files they read at load."""
    from graphify import __main__ as mainmod

    pairs = {
        "_CLAUDE_MD_SECTION": "claude-md",
        "_AGENTS_MD_SECTION": "agents-md",
        "_GEMINI_MD_SECTION": "gemini-md",
        "_VSCODE_INSTRUCTIONS_SECTION": "vscode-instructions",
        "_ANTIGRAVITY_RULES": "antigravity-rules",
        "_KIRO_STEERING": "kiro-steering",
    }
    pkg = Path(mainmod.__file__).parent
    for const_name, basename in pairs.items():
        on_disk = (pkg / "always_on" / f"{basename}.md").read_text(encoding="utf-8")
        assert getattr(mainmod, const_name) == on_disk, const_name


def test_always_on_files_are_guarded_by_check(tmp_path):
    """A hand-edit of an always_on/*.md is caught by --check (the drift guard)."""
    platforms = gen.load_platforms()
    arts = gen.render_all(platforms)
    # The committed + expected/ snapshots match a fresh render.
    assert gen.check(arts) == [], "\n".join(gen.check(arts))
    # A mutated artifact is flagged.
    mutated = [
        gen.RenderedArtifact(a.path, a.content + "drift\n")
        if a.path == "graphify/always_on/claude-md.md"
        else a
        for a in arts
    ]
    problems = gen.check(mutated)
    assert any("always_on/claude-md.md" in p for p in problems)


# --- the per-host coverage audit (the systemic guard) --------------------------


def test_trae_renders_native_agents_md_integration_not_claude():
    """trae wires `graphify trae install` -> AGENTS.md, never `graphify claude install`."""
    core, refs = _platform_artifacts("trae")
    hooks = refs["hooks.md"]
    # The hooks reference carries Trae's native AGENTS.md integration section.
    assert "## For native AGENTS.md integration (Trae)" in hooks
    assert "graphify trae install" in hooks
    assert "graphify trae-cn install" in hooks
    assert "writes a `## graphify` section to the local `AGENTS.md`" in hooks
    # The claude-flavored install command must NOT appear for trae.
    assert "graphify claude install" not in hooks
    assert "native CLAUDE.md integration" not in hooks
    # The lean-core pointer names AGENTS.md, not CLAUDE.md.
    assert "## For the commit hook and native AGENTS.md integration" in core
    assert "wire graphify into a project's AGENTS.md" in core
    assert "native CLAUDE.md integration" not in core


def test_trae_dispatch_carries_the_no_pretooluse_caveat():
    """Trae's B2 dispatch block carries the no-PreToolUse-hook caveat."""
    core, _ = _platform_artifacts("trae")
    b2 = core[core.index("**Step B2") : core.index("Pass the extraction prompt")]
    assert "Trae does NOT support PreToolUse hooks" in b2
    assert "AGENTS.md rules are the always-on mechanism instead" in b2


def test_trae_hooks_reference_includes_the_pretooluse_note():
    """The Trae hooks reference keeps the PreToolUse note in full."""
    _, refs = _platform_artifacts("trae")
    hooks = refs["hooks.md"]
    assert "Unlike Claude Code, Trae does NOT support PreToolUse hooks" in hooks
    assert "Run `/graphify --update` manually after code changes" in hooks


def test_claude_flavored_hosts_keep_their_hooks_text_unchanged():
    """Claude-flavored hosts keep their native hooks text.

    Droid has no Trae caveat and its hooks section names CLAUDE.md; Trae-specific
    behavior must not bleed into Droid or another host.
    """
    for key in ("claude", "droid", "codex", "windows", "kilo", "vscode"):
        core, refs = _platform_artifacts(key)
        hooks = refs["hooks.md"]
        assert "graphify claude install" in hooks, f"[{key}] lost the claude install command"
        assert "native CLAUDE.md integration" in hooks, f"[{key}] lost the CLAUDE.md heading"
        assert "Trae does NOT support PreToolUse hooks" not in core, (
            f"[{key}] leaked the trae caveat"
        )
        assert "Trae does NOT support PreToolUse hooks" not in hooks, (
            f"[{key}] leaked the trae caveat"
        )
        assert "## For the commit hook and native CLAUDE.md integration" in core, (
            f"[{key}] pointer drifted"
        )


# --- the amp native AGENTS.md integration (the 13th split host) ----------------


def test_amp_renders_native_agents_md_integration():
    """Amp wires `graphify amp install` to its native AGENTS.md integration.

    amp shares the agents-md hooks variant with trae but renders its OWN wording:
    a bare "## For native AGENTS.md integration" heading (no "(Trae)" suffix),
    single-line install/uninstall commands (no trae-cn alt), and crucially NO
    PreToolUse caveat.
    """
    core, refs = _platform_artifacts("amp")
    hooks = refs["hooks.md"]
    # Amp's bare heading and host-specific prose.
    assert "## For native AGENTS.md integration" in hooks
    assert "## For native AGENTS.md integration (Trae)" not in hooks
    assert "make graphify always-on in Amp sessions" in hooks
    assert "instructs Amp to check the graph" in hooks
    # amp's single-line install/uninstall, no trae-cn alt comments.
    assert "graphify amp install" in hooks
    assert "graphify amp uninstall  # remove the section" in hooks
    assert "graphify trae install" not in hooks
    assert "graphify trae-cn" not in hooks
    assert "or: graphify" not in hooks
    # No claude flavoring on amp.
    assert "graphify claude install" not in hooks
    assert "native CLAUDE.md integration" not in hooks
    # The lean-core pointer names AGENTS.md, not CLAUDE.md.
    assert "## For the commit hook and native AGENTS.md integration" in core
    assert "wire graphify into a project's AGENTS.md" in core
    assert "native CLAUDE.md integration" not in core


def test_amp_has_no_pretooluse_caveat_anywhere():
    """Amp has no no-PreToolUse-hooks note in either its core or hooks.

    This is the explicit guard against injecting trae-specific wording into amp.
    The caveat belongs to trae alone; amp uses the plain task-tool-disk dispatch
    and a caveat-free AGENTS.md integration section.
    """
    core, refs = _platform_artifacts("amp")
    hooks = refs["hooks.md"]
    assert "PreToolUse" not in core, "amp leaked a PreToolUse caveat into its core"
    assert "PreToolUse" not in hooks, "amp leaked a PreToolUse caveat into its hooks reference"
    assert "Trae does NOT support" not in core
    assert "Trae does NOT support" not in hooks
    # amp's dispatch is the plain task-tool-disk block (no trae caveat line).
    b2 = core[core.index("**Step B2") : core.index("Pass the extraction prompt")]
    assert "Trae" not in b2


def test_agents_renders_its_own_agents_md_hooks_wording():
    """`agents` re-homes amp's agents-md body but with its OWN install wording.

    It shares amp's bare, caveat-free `## For native AGENTS.md integration`
    section (no `(Trae)` suffix, no PreToolUse note) but points at
    `graphify agents install` and is worded for an unspecified host.
    """
    core, refs = _platform_artifacts("agents")
    hooks = refs["hooks.md"]
    assert "## For native AGENTS.md integration" in hooks
    assert "## For native AGENTS.md integration (Trae)" not in hooks
    assert "make graphify always-on in your agent sessions" in hooks
    assert "graphify agents install" in hooks
    assert "graphify agents uninstall  # remove the section" in hooks
    # No amp/trae/claude wording leaks into the agents render.
    assert "graphify amp install" not in hooks
    assert "graphify trae" not in hooks
    assert "graphify claude install" not in hooks
    assert "PreToolUse" not in hooks and "PreToolUse" not in core
    # The lean-core pointer names AGENTS.md, not CLAUDE.md.
    assert "## For the commit hook and native AGENTS.md integration" in core
    assert "native CLAUDE.md integration" not in core


def test_agents_body_matches_amp_modulo_hooks_wording():
    """The agents skill body is amp's body verbatim (it re-homes amp's bundle).

    The two platforms differ only in the hooks reference's install/uninstall
    command wording — everything else (core, query, extraction spec, the other
    six references) is byte-identical.
    """
    platforms = gen.load_platforms()
    amp = {a.path.rsplit("/", 1)[-1]: a.content for a in gen.render(platforms["amp"])}
    agents = {a.path.rsplit("/", 1)[-1]: a.content for a in gen.render(platforms["agents"])}
    # The lean-core skill body is identical (frontmatter + steps, no hooks ref).
    assert amp["skill-amp.md"] == agents["skill-agents.md"]
    # Every reference except hooks.md is byte-identical.
    for name in amp:
        if name in ("skill-amp.md", "hooks.md"):
            continue
        assert amp[name] == agents[name], f"{name} drifted between amp and agents"
    assert amp["hooks.md"] != agents["hooks.md"]


# --- generated query fallback: budget boundaries (executed, not just read) ---

_FALLBACK_BLOCK = re.compile(
    r"^(notice = f'\.\.\. \(truncated.*?^    print\('\\n'\.join\(admitted\).*?$)",
    re.MULTILINE | re.DOTALL,
)

_FIXTURE_LINES = [
    "Traversal: BFS | Start: ['target'] | 4 nodes",
    "  NODE target [src=a.py loc=L1]",
    "  NODE peer [src=b.py loc=L2]",
    "  NODE other [src=c.py loc=L3]",
    "  EDGE target --calls [EXTRACTED]--> peer",
]


# Every generated surface carrying the fallback: 14 platform query references
# plus the Aider and Devin cores, which inline the same block.
EXPECTED_FALLBACK_SURFACES = 16


def _budget_blocks() -> list[tuple[str, str]]:
    """Extract the budget-admission block from every generated surface.

    Reading the text is not enough: the emitted code is what an agent runs, so
    its boundary behaviour has to be executed. Aider and Devin inline the block
    into their cores rather than a references file, so globbing only
    ``skills/*/references/query.md`` silently skipped two real surfaces.
    """
    candidates = sorted(REPO_ROOT.glob("graphify/skills/*/references/query.md"))
    candidates += [REPO_ROOT / "graphify/skill-aider.md", REPO_ROOT / "graphify/skill-devin.md"]
    blocks = []
    for path in sorted(candidates):
        if not path.exists():
            continue
        match = _FALLBACK_BLOCK.search(path.read_text(encoding="utf-8"))
        if match:
            blocks.append((str(path.relative_to(REPO_ROOT)), match.group(1)))
    return blocks


def _run_block(tmp_path, source: str, token_budget: int):
    """Run the generated block the way an agent would: as a real Python process."""
    import json
    import subprocess

    script = tmp_path / "fallback.py"
    prelude = (
        f"token_budget = {token_budget}\n"
        f"char_budget = token_budget * 3\n"
        f"lines = {json.dumps(_FIXTURE_LINES)}\n"
    )
    script.write_text(prelude + source + "\n", encoding="utf-8")
    return subprocess.run(
        [sys.executable, str(script)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=120,
    )


def test_generated_query_fallback_covers_every_surface():
    """Pin the exact surface count.

    Asserting merely "non-empty" let the extractor silently miss the Aider and
    Devin cores, so two shipped surfaces went untested.
    """
    found = sorted(name for name, _ in _budget_blocks())
    assert len(found) == EXPECTED_FALLBACK_SURFACES, found
    assert "graphify/skill-aider.md" in found, found
    assert "graphify/skill-devin.md" in found, found


def _distinct_budget_blocks() -> list[tuple[str, str]]:
    """One representative per DISTINCT block body.

    All surfaces render from the same fragments, so their blocks are usually
    byte-identical. Executing every copy multiplies subprocess cost without
    adding coverage; ``test_generated_query_fallback_covers_every_surface``
    already pins the surface count, and any surface whose block differs shows up
    here as its own representative.
    """
    seen: dict[str, str] = {}
    for name, source in _budget_blocks():
        seen.setdefault(source, name)
    return [(name, source) for source, name in seen.items()]


def test_generated_query_fallback_refuses_budget_below_one_record(tmp_path):
    """A budget too small for one record must refuse, not emit an oversized answer.

    The earlier form always admitted the first line, so token_budget=1 (a
    3-character cap) printed a ~105-character response.
    """
    for name, source in _distinct_budget_blocks():
        result = _run_block(tmp_path, source, token_budget=1)
        assert result.returncode != 0, (name, result.stdout)
        assert not result.stdout.strip(), (name, result.stdout)
        assert "budget" in result.stderr.lower(), (name, result.stderr)


def test_generated_query_fallback_never_exceeds_its_budget(tmp_path):
    """Whatever the generated block emits must fit the budget it declares."""
    for name, source in _distinct_budget_blocks():
        for token_budget in (1, 20, 40, 60, 90, 140, 400):
            result = _run_block(tmp_path, source, token_budget=token_budget)
            if result.returncode != 0:
                continue  # explicit refusal is the correct alternative to overrun
            emitted = result.stdout.rstrip("\n")
            assert len(emitted) <= token_budget * 3, (
                name,
                token_budget,
                len(emitted),
                emitted,
            )


def test_generated_query_fallback_never_returns_header_only_success(tmp_path):
    """Success must carry evidence, not just a header and a truncation notice.

    At a budget that fit the header but no record, the block exited 0 with a
    traversal header plus notice and no NODE/EDGE line at all — apparent success
    conveying nothing.
    """
    for name, source in _distinct_budget_blocks():
        # The failure lives at the boundary where refusal stops, so scan upward
        # and stop after the first few successes rather than sweeping the range.
        successes = 0
        for token_budget in range(1, 120):
            result = _run_block(tmp_path, source, token_budget=token_budget)
            if result.returncode != 0:
                continue
            emitted = result.stdout
            assert any(marker in emitted for marker in ("NODE ", "EDGE ")), (
                f"{name}: budget={token_budget} returned header-only success:\n{emitted}"
            )
            successes += 1
            if successes >= 3:
                break
        assert successes, f"{name}: no budget in 1..119 produced a successful render"
