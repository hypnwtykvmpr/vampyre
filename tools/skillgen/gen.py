"""skillgen: render graphify's committed skill artifacts from edited fragments.

Build-time only. Nothing here ships in the wheel. Fragments under
``tools/skillgen/fragments/`` are the single source of truth a human edits; the
files under ``graphify/skill*.md`` and ``graphify/skills/<platform>/references/``
are generated, committed artifacts. This module renders those artifacts and
guards them against drift.

Usage (from the repo root)::

    python -m tools.skillgen                 # regen every platform's artifacts
    python -m tools.skillgen --platform claude
    python -m tools.skillgen --check         # byte-diff render vs committed + expected/, exit 1 on drift
    python -m tools.skillgen --schema-singleton  # assert the file_type enum is byte-identical everywhere
    python -m tools.skillgen --bless         # rewrite expected/ from the current render

The render is idempotent: the core template's per-platform slots are filled in a
fixed order, the reference index is sorted by name, output is LF-newline, and no
dynamic timestamp or version is written into a generated file. Immutable release
references come from the edited fragments and are guarded against drift.
"""

from __future__ import annotations

import argparse
import re
import sys

try:
    import tomllib  # Python 3.11+ stdlib
except ModuleNotFoundError:  # Python 3.10 - graphify supports >=3.10
    import tomli as tomllib  # type: ignore[no-redef]
from dataclasses import dataclass
from pathlib import Path

from graphify.semantic_schema import render_semantic_schema

# tools/skillgen/gen.py -> repo root is two parents up.
SKILLGEN_DIR = Path(__file__).resolve().parent
REPO_ROOT = SKILLGEN_DIR.parent.parent
FRAGMENTS_DIR = SKILLGEN_DIR / "fragments"
EXPECTED_DIR = SKILLGEN_DIR / "expected"
PLATFORMS_TOML = SKILLGEN_DIR / "platforms.toml"

# The always-on instruction blocks: rendered-file basename -> the __main__.py
# constant it must reproduce. Rendered to graphify/always_on/<basename>.md from
# the matching fragment under fragments/always-on/. These are not platform-
# specific, so they render once in a full run (not under --platform).
ALWAYS_ON_BLOCKS = {
    "claude-md": "_CLAUDE_MD_SECTION",
    "agents-md": "_AGENTS_MD_SECTION",
    "gemini-md": "_GEMINI_MD_SECTION",
    "vscode-instructions": "_VSCODE_INSTRUCTIONS_SECTION",
    "antigravity-rules": "_ANTIGRAVITY_RULES",
    "kiro-steering": "_KIRO_STEERING",
}

# The full six-value file_type enum (Decision A). Every rendered platform — split
# or monolith — must carry exactly this enum, byte for byte. schema-singleton
# guards it.
ENUM_VALUES = "code|document|paper|image|rationale|concept"
ENUM_PROSE = "`code`, `document`, `paper`, `image`, `rationale`, `concept`"

# The eight on-demand references every split platform renders. Six are
# shared-verbatim; two (extraction-spec, hooks) are variant-selected and resolved
# per platform from the extraction/hooks_variant fields.
_SHARED_REFERENCES = {
    "update": "references/shared/update.md",
    "exports": "references/shared/exports.md",
    "github-and-merge": "references/shared/github-and-merge.md",
    "transcribe": "references/shared/transcribe.md",
    "add-watch": "references/shared/add-watch.md",
}
_EXTRACTION_SOURCE = {
    "verbose": "references/shared/extraction-spec.md",
    "compact": "references/shared/extraction-spec-compact.md",
}
# Single unified query reference + stub: superior vocab-expansion (Step 0) plus
# CLI traversal plus inline NetworkX fallback, shipped to every platform. The
# capabilities used to be split across cli.md / cli-inline.md so no platform got
# both — Claude had expansion but no fallback, the rest had the fallback but the
# weaker matcher (#1325).
_QUERY_REFERENCE = "references/query/default.md"
_QUERY_STUB = "query-stub/default.md"
# The hooks reference is host-flavored. Most hosts read CLAUDE.md and wire
# always-on via `graphify claude install` (the shared body). The agents-md hosts
# (trae, trae-cn, amp) read AGENTS.md and wire it via `graphify <host> install`.
# The agents-md fragment is a per-host template: the install/uninstall commands,
# the host display name, the heading suffix, and the PreToolUse caveat are slots
# filled from _AGENTS_MD_HOOKS per host. Trae carries the caveat that it does not
# support PreToolUse hooks; Amp has no such caveat, so its slot is empty.
# Each variant also drives the @@HOOKS_TARGET@@ pointer text in the core. The
# variant key matches the prose target file the pointer names.
_HOOKS_SOURCE = {
    "claude-md": "references/shared/hooks.md",
    "agents-md": "references/host/hooks-agents-md.md",
}

# Per-host slots for the agents-md hooks reference template. Rendered EXACTLY as
# that host's native skill body uses for "## For native AGENTS.md integration".
# Trae carries a heading suffix, alternate commands, and the no-PreToolUse note;
# Amp has a bare heading, one command pair, and no caveat.
_TRAE_PRETOOLUSE_NOTE = (
    "\n> **Note:** Unlike Claude Code, Trae does NOT support PreToolUse hooks. "
    "The AGENTS.md rules are the always-on mechanism — there is no automatic graph "
    "rebuild on tool use. Run `/graphify --update` manually after code changes if "
    "the graph needs refreshing.\n"
)
_AGENTS_MD_HOOKS: dict[str, dict[str, str]] = {
    "trae": {
        "heading_suffix": " (Trae)",
        "host_display": "Trae",
        "install_block": "graphify trae install       # or: graphify trae-cn install",
        "uninstall_block": "graphify trae uninstall     # or: graphify trae-cn uninstall   # remove the section",
        "pretooluse_note": _TRAE_PRETOOLUSE_NOTE,
    },
    "amp": {
        "heading_suffix": "",
        "host_display": "Amp",
        "install_block": "graphify amp install",
        "uninstall_block": "graphify amp uninstall  # remove the section",
        "pretooluse_note": "",
    },
    "agents": {
        # The generic cross-framework Agent-Skills target. Mirrors amp's bare,
        # caveat-free agents-md section, worded for an unspecified host and
        # pointing at `graphify agents install` (which wires AGENTS.md, like amp).
        "heading_suffix": "",
        "host_display": "your agent",
        "install_block": "graphify agents install",
        "uninstall_block": "graphify agents uninstall  # remove the section",
        "pretooluse_note": "",
    },
}
# The prose file name the lean-core hooks pointer names, per hooks variant.
_HOOKS_TARGET = {
    "claude-md": "CLAUDE.md",
    "agents-md": "AGENTS.md",
}


@dataclass(frozen=True)
class Platform:
    """One render unit parsed from platforms.toml."""

    key: str
    bucket: str
    skill_dst: str
    # split-only template inputs
    core: str | None = None
    refs_dst: str | None = None
    name: str = "graphify"
    description: str | None = None
    trigger: str | None = None  # removed — not part of Agent Skills spec (#1180)
    dispatch: str | None = None
    extraction: str = "verbose"
    shell: str = "posix"
    claude_md: bool = False
    hooks_variant: str = "claude-md"
    extra_sections: tuple[str, ...] = ()
    # monolith-only inputs
    monolith: str | None = None

    def reference_sources(self) -> dict[str, str]:
        """Resolve the rendered-name -> source-fragment map for this split platform."""
        refs = dict(_SHARED_REFERENCES)
        refs["extraction-spec"] = _EXTRACTION_SOURCE[self.extraction]
        refs["query"] = _QUERY_REFERENCE
        refs["hooks"] = _HOOKS_SOURCE[self.hooks_variant]
        return refs

    @property
    def hooks_target(self) -> str:
        """The prose file name the lean-core hooks pointer names for this host."""
        return _HOOKS_TARGET[self.hooks_variant]


def load_platforms() -> dict[str, Platform]:
    """Parse platforms.toml into Platform records, keyed by platform name."""
    data = tomllib.loads(PLATFORMS_TOML.read_text(encoding="utf-8"))
    out: dict[str, Platform] = {}
    for key, cfg in data.get("platform", {}).items():
        out[key] = Platform(
            key=key,
            bucket=cfg["bucket"],
            skill_dst=cfg["skill_dst"],
            core=cfg.get("core"),
            refs_dst=cfg.get("refs_dst"),
            name=cfg.get("name", "graphify"),
            description=cfg.get("description"),
            trigger=cfg.get("trigger"),
            dispatch=cfg.get("dispatch"),
            extraction=cfg.get("extraction", "verbose"),
            shell=cfg.get("shell", "posix"),
            claude_md=bool(cfg.get("claude_md", False)),
            hooks_variant=cfg.get("hooks_variant", "claude-md"),
            extra_sections=tuple(cfg.get("extra_sections", [])),
            monolith=cfg.get("monolith"),
        )
    return out


def _read_fragment(rel: str) -> str:
    """Read a fragment file under fragments/, normalised to LF newlines."""
    text = (FRAGMENTS_DIR / rel).read_text(encoding="utf-8")
    return _normalise(text)


def _normalise(text: str) -> str:
    """Force LF newlines and exactly one trailing newline."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return text.rstrip("\n") + "\n"


@dataclass(frozen=True)
class RenderedArtifact:
    """A single generated file: its repo-relative path and exact bytes."""

    path: str  # relative to REPO_ROOT
    content: str


def _render_frontmatter(platform: Platform) -> str:
    """Render the YAML frontmatter from the platform's name and description.

    Only emits fields from the Agent Skills spec (name, description).
    The description is preserved verbatim from platforms.toml — never invented.
    """
    if platform.description is None:
        raise ValueError(f"split platform '{platform.key}' is missing a description")
    lines = ["---", f"name: {platform.name}", f'description: "{platform.description}"']
    lines.append("---")
    return "\n".join(lines)


def _render_core(platform: Platform) -> str:
    """Fill the shared core template's per-platform slots for this platform."""
    template = _read_fragment(f"core/{platform.core}.md")

    if platform.dispatch is None:
        raise ValueError(f"split platform '{platform.key}' is missing a dispatch variant")

    install = _read_fragment(f"shell/{platform.shell}.md").rstrip("\n")
    dispatch = _read_fragment(f"dispatch/{platform.dispatch}.md").rstrip("\n")
    query_stub = _read_fragment(_QUERY_STUB).rstrip("\n")

    if platform.extra_sections:
        extra = "".join(
            _read_fragment(f"extra/{name}.md").rstrip("\n") + "\n\n"
            for name in platform.extra_sections
        )
    else:
        extra = ""

    body = (
        template.replace("@@FRONTMATTER@@", _render_frontmatter(platform))
        .replace("@@INSTALL@@", install)
        .replace("@@DISPATCH@@", dispatch)
        .replace("@@QUERY_STUB@@", query_stub)
        .replace("@@HOOKS_TARGET@@", platform.hooks_target)
        .replace("@@EXTRA@@", extra)
    )
    if "@@" in body:
        leftover = sorted(set(re.findall(r"@@\w+@@", body)))
        raise ValueError(f"unfilled core slots for '{platform.key}': {leftover}")
    return _normalise(body)


def _render_agents_md_hooks(platform: Platform) -> str:
    """Fill the agents-md hooks template's per-host slots for this platform.

    The fragment is one template shared by every AGENTS.md host (trae, trae-cn,
    amp). The install/uninstall commands, the host display name, the heading
    suffix, and the PreToolUse caveat are filled from _AGENTS_MD_HOOKS so each host
    renders its own wording: Trae keeps the "(Trae)" heading suffix and the
    no-PreToolUse note; Amp gets a bare heading, single-line commands, and no
    caveat.
    """
    template = _read_fragment(_HOOKS_SOURCE["agents-md"])
    slots = _AGENTS_MD_HOOKS.get(platform.key)
    if slots is None:
        raise ValueError(
            f"platform '{platform.key}' uses the agents-md hooks variant but has no "
            f"_AGENTS_MD_HOOKS entry"
        )
    body = (
        template.replace("@@AGENTS_HEADING_SUFFIX@@", slots["heading_suffix"])
        .replace("@@HOST_DISPLAY@@", slots["host_display"])
        .replace("@@AGENTS_INSTALL_BLOCK@@", slots["install_block"])
        .replace("@@AGENTS_UNINSTALL_BLOCK@@", slots["uninstall_block"])
        .replace("@@AGENTS_PRETOOLUSE_NOTE@@", slots["pretooluse_note"])
    )
    if "@@" in body:
        leftover = sorted(set(re.findall(r"@@\w+@@", body)))
        raise ValueError(f"unfilled agents-md hooks slots for '{platform.key}': {leftover}")
    return _normalise(body)


def render(platform: Platform) -> list[RenderedArtifact]:
    """Render every committed artifact for one platform.

    A split platform yields the lean core SKILL.md plus one file per reference,
    in a stable order (core first, then references sorted by name). A monolith
    yields a single inline skill body.
    """
    if platform.bucket == "monolith":
        body = _read_fragment(f"core/{platform.monolith}.md")
        safe_update = _read_fragment("references/shared/monolith-update.md").rstrip()
        pattern = (
            r"## For --update \(incremental re-extraction\)\n.*?"
            r"\n---\n\n## For --cluster-only"
        )
        body, replacements = re.subn(
            pattern,
            safe_update + "\n\n---\n\n## For --cluster-only",
            body,
            flags=re.DOTALL,
        )
        if replacements != 1:
            raise ValueError(f"monolith '{platform.key}' must contain exactly one update section")
        return [RenderedArtifact(platform.skill_dst, body)]

    if platform.bucket != "split":
        raise ValueError(f"unknown bucket '{platform.bucket}' for platform '{platform.key}'")

    if platform.refs_dst is None:
        raise ValueError(f"split platform '{platform.key}' is missing refs_dst")

    artifacts: list[RenderedArtifact] = [
        RenderedArtifact(platform.skill_dst, _render_core(platform))
    ]

    references = platform.reference_sources()
    # Sorted reference index keeps the output idempotent regardless of map order.
    for name in sorted(references):
        # The agents-md hooks reference is a per-host template; everything else is
        # read verbatim.
        if name == "hooks" and platform.hooks_variant == "agents-md":
            body = _render_agents_md_hooks(platform)
        else:
            body = _read_fragment(references[name])
        if name == "extraction-spec":
            marker = "@@SEMANTIC_SCHEMA@@"
            if body.count(marker) != 1:
                raise ValueError(
                    f"extraction schema template '{references[name]}' must contain one {marker}"
                )
            body = body.replace(marker, render_semantic_schema())
        rel = f"{platform.refs_dst}/{name}.md"
        artifacts.append(RenderedArtifact(rel, body))
    return artifacts


def render_always_on() -> list[RenderedArtifact]:
    """Render the six always-on instruction blocks to graphify/always_on/*.md.

    These are the blocks the installer injects into shared files (CLAUDE.md,
    AGENTS.md, GEMINI.md, .github/copilot-instructions.md, Antigravity rules,
    Kiro steering). They used to be triple-quoted constants in __main__.py and
    are now packaged markdown the module reads at load. Rendering them through
    skillgen puts them under the --check / expected/ drift guard like every other
    generated artifact. They are not platform-specific, so they render once.
    """
    out: list[RenderedArtifact] = []
    for basename in sorted(ALWAYS_ON_BLOCKS):
        body = _read_fragment(f"always-on/{basename}.md")
        out.append(RenderedArtifact(f"graphify/always_on/{basename}.md", body))
    return out


def render_all(platforms: dict[str, Platform], only: str | None = None) -> list[RenderedArtifact]:
    """Render the selected platforms (or all), flattened into one artifact list.

    A full render (no ``only``) also includes the always-on blocks; a single
    ``--platform`` render does not, since the always-on files are shared, not
    per-platform.
    """
    keys = [only] if only else sorted(platforms)
    out: list[RenderedArtifact] = []
    for key in keys:
        if key not in platforms:
            raise SystemExit(
                f"error: unknown platform '{key}'. Known: {', '.join(sorted(platforms))}"
            )
        out.extend(render(platforms[key]))
    if only is None:
        out.extend(render_always_on())
    return out


def write_artifacts(artifacts: list[RenderedArtifact]) -> list[str]:
    """Write artifacts to disk under REPO_ROOT. Returns the paths written."""
    written: list[str] = []
    for art in artifacts:
        dst = REPO_ROOT / art.path
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(art.content, encoding="utf-8", newline="\n")
        written.append(art.path)
    return written


def _expected_path(rel: str) -> Path:
    """Map a repo-relative artifact path to its expected/ snapshot path.

    The artifact path is flattened (``/`` -> ``__``) into a single filename so
    the snapshot tree never contains a ``skills/`` path component, which the
    repo .gitignore ignores. This keeps expected/ a flat, fully tracked dir.
    """
    return EXPECTED_DIR / (rel.replace("/", "__"))


def bless(artifacts: list[RenderedArtifact]) -> list[str]:
    """Write the current render into expected/ as the blessed snapshot."""
    written: list[str] = []
    for art in artifacts:
        dst = _expected_path(art.path)
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(art.content, encoding="utf-8", newline="\n")
        written.append(str(dst.relative_to(SKILLGEN_DIR)))
    return written


def check(artifacts: list[RenderedArtifact]) -> list[str]:
    """Byte-diff the render against both committed artifacts and expected/.

    Returns a list of human-readable drift messages. Empty list means clean.
    This is the anti-drift guard wired into CI and pre-commit: any hand-edit of
    a generated file, or a stale expected/ snapshot, is caught here.
    """
    problems: list[str] = []
    for art in artifacts:
        committed = REPO_ROOT / art.path
        if not committed.exists():
            problems.append(
                f"missing committed artifact: {art.path} (run: python -m tools.skillgen)"
            )
        elif committed.read_text(encoding="utf-8") != art.content:
            problems.append(
                f"committed artifact out of date: {art.path} (run: python -m tools.skillgen)"
            )

        snapshot = _expected_path(art.path)
        if not snapshot.exists():
            problems.append(
                f"missing expected/ snapshot: {art.path} (run: python -m tools.skillgen --bless)"
            )
        elif snapshot.read_text(encoding="utf-8") != art.content:
            problems.append(
                f"expected/ snapshot out of date: {art.path} (run: python -m tools.skillgen --bless)"
            )
    return problems


def headings(markdown: str) -> list[str]:
    """Return the ATX markdown headings in source order, ignoring code fences.

    A ``#``-prefixed line inside a fenced code block is a shell comment, not a
    heading, so fence state is tracked to avoid counting them.
    """
    out: list[str] = []
    in_fence = False
    fence_marker = ""
    for line in markdown.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("```") or stripped.startswith("~~~"):
            marker = stripped[:3]
            if not in_fence:
                in_fence = True
                fence_marker = marker
            elif marker == fence_marker:
                in_fence = False
                fence_marker = ""
            continue
        if in_fence:
            continue
        # An ATX heading is 1-6 '#' then a space then text.
        if stripped.startswith("#"):
            hashes = len(stripped) - len(stripped.lstrip("#"))
            if 1 <= hashes <= 6 and stripped[hashes : hashes + 1] == " ":
                out.append(stripped.strip())
    return out


def _enum_lines(content: str) -> list[str]:
    """Return every line in a rendered artifact that carries the file_type enum."""
    return [line for line in content.splitlines() if ENUM_VALUES in line or ENUM_PROSE in line]


# Legacy enum fragments that must never survive the six-value unification. Each
# is a strict prefix of the full superset, so a line carrying one WITHOUT the
# full superset is a stale 4- or 5-value enum.
_LEGACY_ENUMS = (
    "code|document|paper|image|rationale",  # 5-value
    "code|document|paper|image",  # 4-value
)


def legacy_enum_lines(content: str) -> list[str]:
    """Return lines carrying a legacy (sub-superset) file_type enum.

    A line counts as legacy only when it has a 4- or 5-value enum fragment but
    NOT the full six-value superset. The schema-singleton guard treats any such
    line as drift.
    """
    out: list[str] = []
    for line in content.splitlines():
        if ENUM_VALUES in line:
            continue
        if any(bad in line for bad in _LEGACY_ENUMS):
            out.append(line.strip())
    return out


def schema_singleton(platforms: dict[str, Platform]) -> list[str]:
    """Assert the file_type enum block is byte-identical across every platform.

    Every rendered artifact that mentions the enum — the verbose and compact
    extraction specs, and the inline monolith bodies — must carry exactly the
    six-value superset and nothing else. A stray 4- or 5-value enum line is the
    failure this guard exists to catch.
    """
    problems: list[str] = []
    for key in sorted(platforms):
        for art in render(platforms[key]):
            for stripped in legacy_enum_lines(art.content):
                problems.append(
                    f"[{key}] {art.path}: legacy file_type enum (not the six-value superset): {stripped!r}"
                )
    return problems


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python -m tools.skillgen",
        description="Render and guard graphify's committed skill artifacts.",
    )
    p.add_argument("--platform", help="render or check just this platform key")
    p.add_argument(
        "--check",
        action="store_true",
        help="byte-diff render vs committed + expected/, exit 1 on drift",
    )
    p.add_argument(
        "--schema-singleton",
        action="store_true",
        help="assert the file_type enum is byte-identical everywhere",
    )
    p.add_argument("--bless", action="store_true", help="rewrite expected/ from the current render")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    platforms = load_platforms()

    if args.schema_singleton:
        problems = schema_singleton(
            {args.platform: platforms[args.platform]} if args.platform else platforms
        )
        if problems:
            print("schema-singleton FAILED (file_type enum drift):", file=sys.stderr)
            for m in problems:
                print(f"  {m}", file=sys.stderr)
            return 1
        print("schema-singleton OK: the file_type enum is the six-value superset everywhere.")
        return 0

    artifacts = render_all(platforms, only=args.platform)

    if args.check:
        problems = check(artifacts)
        if problems:
            print("check FAILED (skill artifacts have drifted):", file=sys.stderr)
            for m in problems:
                print(f"  {m}", file=sys.stderr)
            return 1
        print(f"check OK: {len(artifacts)} artifact(s) match committed output and expected/.")
        return 0

    if args.bless:
        written = bless(artifacts)
        print(f"blessed {len(written)} artifact(s) into expected/.")
        return 0

    written = write_artifacts(artifacts)
    print(f"rendered {len(written)} artifact(s):")
    for path in written:
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
