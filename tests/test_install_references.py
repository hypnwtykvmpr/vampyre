"""Tests for the progressive-disclosure references/ sidecar install path.

The real claude bundle now ships in the package (graphify/skills/claude/), so
claude and its reuse twins (antigravity, kimi) install progressively: a lean
SKILL.md plus a references/ sidecar. Every other host whose bundle has not
shipped yet still installs today's byte-identical monolith.

The plumbing tests below stage a hand-made fake bundle in claude's slot so the
dir-copy, version-stamp, reinstall, and uninstall flow can be exercised with
fixed, asserted content. The fixture backs up the real committed bundle and
restores it on teardown, so the working tree is never disturbed.
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

import graphify
import graphify.__main__ as mainmod


PKG_DIR = Path(graphify.__file__).parent


@pytest.fixture()
def fake_bundle(tmp_path, monkeypatch):
    """Route claude to an isolated packaged references bundle."""
    platform = "claude"
    refs_dir = tmp_path / "packaged" / "claude" / "references"
    refs_dir.mkdir(parents=True)
    (refs_dir / "extraction-spec.md").write_text("# extraction spec fragment\n", encoding="utf-8")
    (refs_dir / "query.md").write_text("# query fragment\n", encoding="utf-8")
    real_resolver = mainmod._packaged_skill_refs_dir
    monkeypatch.setattr(
        mainmod,
        "_packaged_skill_refs_dir",
        lambda name: refs_dir if name == platform else real_resolver(name),
    )
    return platform


def _install(tmp_path, platform):
    old_cwd = Path.cwd()
    try:
        os.chdir(tmp_path)
        platform_env = {
            "HOME": str(tmp_path),
            "USERPROFILE": str(tmp_path),
            "LOCALAPPDATA": str(tmp_path / "AppData" / "Local"),
            "APPDATA": str(tmp_path / "AppData" / "Roaming"),
            "XDG_CONFIG_HOME": str(tmp_path / ".config"),
        }
        with (
            patch.dict(os.environ, platform_env),
            patch("graphify.__main__.Path.home", return_value=tmp_path),
        ):
            mainmod.install(platform=platform)
            return mainmod._platform_skill_destination(platform)
    finally:
        os.chdir(old_cwd)


def test_install_stages_references_sidecar(tmp_path, fake_bundle):
    """A progressive platform install drops references/ alongside SKILL.md."""
    platform = fake_bundle
    _install(tmp_path, platform)
    skill_dir = tmp_path / ".claude" / "skills" / "graphify"
    assert (skill_dir / "SKILL.md").exists()
    refs = skill_dir / "references"
    assert refs.is_dir()
    assert (refs / "extraction-spec.md").read_text(
        encoding="utf-8"
    ) == "# extraction spec fragment\n"
    assert (refs / "query.md").read_text(encoding="utf-8") == "# query fragment\n"
    # No leftover staging dir.
    assert not (skill_dir / "references.tmp").exists()


def test_single_version_stamp_covers_skill_and_references(tmp_path, fake_bundle):
    """One .graphify_version stamp versions SKILL.md + references/ together."""
    platform = fake_bundle
    _install(tmp_path, platform)
    skill_dir = tmp_path / ".claude" / "skills" / "graphify"
    stamps = list(skill_dir.rglob(".graphify_version"))
    assert len(stamps) == 1
    assert stamps[0] == skill_dir / ".graphify_version"
    assert stamps[0].read_text(encoding="utf-8") == mainmod.__version__


def test_reinstall_replaces_references_atomically(tmp_path, fake_bundle):
    """Reinstall swaps references/ in place, dropping a stale fragment."""
    platform = fake_bundle
    _install(tmp_path, platform)
    skill_dir = tmp_path / ".claude" / "skills" / "graphify"
    refs = skill_dir / "references"
    # Simulate a stale fragment left from an older install.
    (refs / "stale-old.md").write_text("stale\n", encoding="utf-8")
    assert (refs / "stale-old.md").exists()

    _install(tmp_path, platform)

    # The stale fragment is gone; the packaged ones are present.
    assert not (refs / "stale-old.md").exists()
    assert (refs / "extraction-spec.md").exists()
    assert (refs / "query.md").exists()
    assert not (skill_dir / "references.tmp").exists()


def test_uninstall_removes_references_then_walks_dirs(tmp_path, fake_bundle):
    """Uninstall rmtrees references/ before the dir walk so the tree is cleared."""
    platform = fake_bundle
    _install(tmp_path, platform)
    skill_dir = tmp_path / ".claude" / "skills" / "graphify"
    assert (skill_dir / "references").is_dir()

    with patch("graphify.__main__.Path.home", return_value=tmp_path):
        removed = mainmod._remove_skill_file(platform)

    assert removed
    assert not skill_dir.exists()
    # The 3-level walk collapsed the now-empty skill dirs.
    assert not (tmp_path / ".claude" / "skills").exists()


def test_check_skill_version_warns_on_missing_references(tmp_path, fake_bundle, capsys):
    """If SKILL.md links references/ but the dir is gone, warn to repair."""
    platform = fake_bundle
    _install(tmp_path, platform)
    skill_dir = tmp_path / ".claude" / "skills" / "graphify"
    skill = skill_dir / "SKILL.md"
    # Force the body to reference the sidecar, then delete the sidecar.
    skill.write_text("See references/extraction-spec.md for the schema.\n", encoding="utf-8")
    shutil.rmtree(skill_dir / "references")

    mainmod._check_skill_version(skill)

    err = capsys.readouterr().err
    assert "references/ sidecar is missing" in err


def test_check_skill_version_ignores_permission_error(tmp_path, fake_bundle, monkeypatch, capsys):
    """Unreadable version probes should not crash startup."""
    platform = fake_bundle
    _install(tmp_path, platform)
    skill_dir = tmp_path / ".claude" / "skills" / "graphify"
    skill = skill_dir / "SKILL.md"

    # Drain install output so the no-warning assertions below see only what
    # _check_skill_version itself emits.
    capsys.readouterr()

    original_exists = Path.exists

    def guarded_exists(self):
        if self.name == ".graphify_version":
            raise PermissionError("denied")
        return original_exists(self)

    monkeypatch.setattr(Path, "exists", guarded_exists)

    mainmod._check_skill_version(skill)

    out = capsys.readouterr()
    assert out.out == ""
    assert out.err == ""


def test_hard_fail_when_bundle_dir_present_but_references_missing(tmp_path, monkeypatch):
    """A bundle dir that exists but has no references/ subdir is a malformed
    package: exit 1 rather than silently shipping an empty sidecar.

    A wholly-absent bundle dir is the opposite case (a host whose wave has not
    shipped) and falls back to monolith, covered by
    ``test_unbuilt_bundle_host_falls_back_to_monolith``.
    """
    bundle_dir = tmp_path / "packaged" / "claude"
    # Bundle dir present, but no references/ subdir inside it.
    bundle_dir.mkdir(parents=True, exist_ok=True)
    (bundle_dir / "SKILL.md").write_text("body\n", encoding="utf-8")
    monkeypatch.setattr(
        mainmod,
        "_packaged_skill_refs_dir",
        lambda _name: bundle_dir / "references",
    )

    with pytest.raises(SystemExit) as exc:
        with patch("graphify.__main__.Path.home", return_value=tmp_path):
            monkeypatch.chdir(tmp_path)
            mainmod._copy_skill_file("claude")
    assert exc.value.code == 1


def test_unbuilt_bundle_host_falls_back_to_monolith(tmp_path, monkeypatch):
    """A progressive host whose bundle is absent installs its monolith."""
    host = "claude"
    monolith = PKG_DIR / "skill.md"
    missing_bundle = "__test_missing_bundle__"
    assert not (PKG_DIR / "skills" / missing_bundle).exists()
    monkeypatch.setitem(mainmod._PLATFORM_CONFIG[host], "skill_refs", missing_bundle)
    _install(tmp_path, host)
    with patch("graphify.__main__.Path.home", return_value=tmp_path):
        dst = mainmod._platform_skill_destination(host)
    skill_dir = dst.parent
    assert (skill_dir / "SKILL.md").exists()
    assert not (skill_dir / "references").exists()
    # Byte-identical to the packaged monolith for that host.
    assert (skill_dir / "SKILL.md").read_bytes() == monolith.read_bytes()


def test_claude_install_ships_lean_core_and_references(tmp_path):
    """claude's real bundle ships: a lean SKILL.md plus the references/ sidecar."""
    skills_claude = PKG_DIR / "skills" / "claude" / "references"
    assert skills_claude.is_dir(), "claude bundle must ship in this build"
    _install(tmp_path, "claude")
    skill_dir = tmp_path / ".claude" / "skills" / "graphify"
    skill = skill_dir / "SKILL.md"
    refs = skill_dir / "references"
    assert skill.exists()
    assert refs.is_dir()
    body = skill.read_text(encoding="utf-8")
    # The installed SKILL.md is exactly the packaged lean core (skill.md IS the
    # lean core now, not the old monolith).
    assert body == (PKG_DIR / "skill.md").read_text(encoding="utf-8")
    # The lean core points at its references and dropped the on-demand content.
    assert "references/extraction-spec.md" in body
    # The full embedded subagent prompt was moved out into the references.
    assert '"file_type":"code|document|paper|image|rationale|concept"' not in body
    # The lean core is materially smaller than the old ~1156-line monolith.
    assert len(body.splitlines()) < 800
    # The version stamp covers SKILL.md + references/ together.
    assert (skill_dir / ".graphify_version").read_text(encoding="utf-8") == mainmod.__version__
    # The eight on-demand fragments all landed.
    names = sorted(p.name for p in refs.glob("*.md"))
    assert names == [
        "add-watch.md",
        "exports.md",
        "extraction-spec.md",
        "github-and-merge.md",
        "hooks.md",
        "query.md",
        "transcribe.md",
        "update.md",
    ]


def test_gemini_install_references_all_resolve(tmp_path):
    """End-to-end: every references/ pointer in gemini's installed SKILL.md resolves.

    gemini ships claude's lean skill.md body but resolves its references through a
    separate path, so this locks the body<->refs coupling: a real install with the
    real claude bundle must leave no dead pointer on disk.
    """
    import re

    skill = _install(tmp_path, "gemini")
    assert skill.exists()
    refdir = skill.parent / "references"
    assert refdir.is_dir()
    pointers = set(re.findall(r"references/([a-z-]+\.md)", skill.read_text(encoding="utf-8")))
    assert pointers, "the lean core should link to references/"
    missing = [p for p in pointers if not (refdir / p).is_file()]
    assert not missing, f"dead reference pointers in gemini install: {missing}"


def test_claude_twins_ride_the_claude_bundle(tmp_path):
    """antigravity and kimi reuse claude's split bundle, so they go progressive too.

    gemini has no _PLATFORM_CONFIG entry but installs claude's skill.md body
    verbatim; that body is the lean progressive core, so gemini must ride the
    claude references bundle as well or it ships a SKILL.md with dead pointers.
    """
    for platform in ("antigravity", "kimi", "gemini"):
        refs_src = mainmod._packaged_skill_refs_dir(platform)
        assert refs_src is not None
        assert refs_src == PKG_DIR / "skills" / "claude" / "references"


def test_pyproject_declares_references_globs():
    """package-data must declare the references + always-on globs that ship the bundles.

    The references sidecar ships under graphify/skills/<host>/references/ and the
    always-on injection blocks under graphify/always_on/. The earlier
    skills/*/SKILL.md glob matched nothing (no graphify/skills/<host>/SKILL.md
    exists; the skill bodies are graphify/skill*.md, listed explicitly), so it was
    removed. This test guards the real shipped layout.
    """
    try:
        import tomllib  # Python 3.11+ stdlib
    except ModuleNotFoundError:  # Python 3.10
        import tomli as tomllib  # type: ignore[no-redef]

    pyproject = PKG_DIR.parent / "pyproject.toml"
    assert pyproject.exists(), "repository packaging tests require pyproject.toml"
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    pkg_data = data["tool"]["setuptools"]["package-data"]["graphify"]
    assert "skills/*/references/*.md" in pkg_data
    assert "always_on/*.md" in pkg_data
    # The dead glob that matched no file must not creep back in.
    assert "skills/*/SKILL.md" not in pkg_data


# The full progressive-disclosure payload the wheel must ship: 15 skill bodies,
# 104 references (13 split hosts x 8 each), and 6 always-on injection blocks.
_EXPECTED_SKILL_BODIES = (
    "skill.md",
    "skill-codex.md",
    "skill-opencode.md",
    "skill-kilo.md",
    "skill-aider.md",
    "skill-amp.md",
    "skill-copilot.md",
    "skill-claw.md",
    "skill-windows.md",
    "skill-droid.md",
    "skill-trae.md",
    "skill-kiro.md",
    "skill-vscode.md",
    "skill-pi.md",
    "skill-devin.md",
)
_SPLIT_HOSTS = (
    "claude",
    "codex",
    "windows",
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
_REFERENCE_NAMES = (
    "add-watch.md",
    "exports.md",
    "extraction-spec.md",
    "github-and-merge.md",
    "hooks.md",
    "query.md",
    "transcribe.md",
    "update.md",
)
_ALWAYS_ON_NAMES = (
    "agents-md.md",
    "antigravity-rules.md",
    "claude-md.md",
    "gemini-md.md",
    "kiro-steering.md",
    "vscode-instructions.md",
)


def _build_wheel_names(repo_root):
    """Build the wheel and return the set of arcnames inside it.

    Fails loudly (not skip) when the build backend is unavailable: `build` is a
    declared dev dependency, so an environment that runs this test is expected to
    have it. A silent skip is how a packaging regression slips through CI.
    """
    import subprocess
    import sys
    import tempfile
    import zipfile

    try:
        import build  # noqa: F401
    except ImportError:
        raise AssertionError(
            "the 'build' module is required for the wheel-content test but is not "
            "installed; it is a declared dev dependency (run `uv sync --all-extras`)"
        )

    with tempfile.TemporaryDirectory() as outdir:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "build",
                "--wheel",
                "--no-isolation",
                "--outdir",
                outdir,
                str(repo_root),
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        assert result.returncode == 0, (
            "wheel build failed:\n"
            f"stdout:\n{result.stdout[-1000:]}\n"
            f"stderr:\n{result.stderr[-1000:]}"
        )
        wheels = list(Path(outdir).glob("*.whl"))
        assert wheels, "no wheel was produced"
        with zipfile.ZipFile(wheels[0]) as zf:
            return set(zf.namelist())


def test_built_wheel_ships_the_full_skill_payload():
    """The built wheel must carry every skill body, reference, and always-on block.

    This is the headline regression guard. If the package-data globs fail to match
    (e.g. the stale skills/*/SKILL.md glob that matched nothing), the wheel ships a
    SKILL.md with no references/ sidecar and an install silently loses every
    on-demand fragment. The test asserts the whole shipped layout: 15 skill
    bodies, 96 references, and 6 always-on injection blocks. It FAILS (not skips)
    when the build backend is missing, because build is a declared dev dependency.
    """
    repo_root = PKG_DIR.parent
    assert (repo_root / "pyproject.toml").exists(), (
        "repository packaging tests require pyproject.toml"
    )
    # In this wave every split-host bundle ships, so its absence is a real failure,
    # not a reason to skip.
    assert (PKG_DIR / "skills" / "claude" / "references" / "extraction-spec.md").exists(), (
        "the claude bundle must ship in this build; the references sidecar is missing"
    )

    names = _build_wheel_names(repo_root)

    missing_bodies = [b for b in _EXPECTED_SKILL_BODIES if f"graphify/{b}" not in names]
    assert not missing_bodies, f"wheel is missing skill bodies: {missing_bodies}"
    assert len(_EXPECTED_SKILL_BODIES) == 15

    missing_refs = [
        f"graphify/skills/{host}/references/{ref}"
        for host in _SPLIT_HOSTS
        for ref in _REFERENCE_NAMES
        if f"graphify/skills/{host}/references/{ref}" not in names
    ]
    assert not missing_refs, f"wheel is missing references: {missing_refs}"
    assert len(_SPLIT_HOSTS) * len(_REFERENCE_NAMES) == 104

    missing_always_on = [
        f"graphify/always_on/{name}"
        for name in _ALWAYS_ON_NAMES
        if f"graphify/always_on/{name}" not in names
    ]
    assert not missing_always_on, f"wheel is missing always-on blocks: {missing_always_on}"
    assert len(_ALWAYS_ON_NAMES) == 6

    # The specific headline file that the stale glob would have dropped.
    assert "graphify/skills/claude/references/extraction-spec.md" in names
    assert "graphify/skills/trae/references/hooks.md" in names
    # amp is now a split host too; its bundle must ship like every other.
    assert "graphify/skills/amp/references/hooks.md" in names


def test_monolith_install_clears_orphan_references(tmp_path, fake_bundle):
    """A monolith platform install removes any orphan references/ left behind."""
    # aider is a monolith (no skill_refs). Seed an orphan references/ dir at its
    # destination, then install and confirm it is cleared.
    skill_dst = tmp_path / ".aider" / "graphify" / "SKILL.md"
    orphan = skill_dst.parent / "references"
    orphan.mkdir(parents=True)
    (orphan / "leftover.md").write_text("leftover\n", encoding="utf-8")
    _install(tmp_path, "aider")
    assert skill_dst.exists()
    assert not orphan.exists()


def test_amp_user_install_carries_references(tmp_path, monkeypatch):
    """amp is progressive: its corrected user dir also gets the references/ sidecar.

    amp's real bundle now ships in the package (graphify/skills/amp/), so the
    install copies the actual committed references alongside SKILL.md. This is the
    case the progressive split was built to cover: amp was the omitted 13th host.
    """
    from graphify.__main__ import main

    assert (PKG_DIR / "skills" / "amp" / "references" / "hooks.md").exists(), (
        "amp's references bundle must ship in this build"
    )

    home = tmp_path / "home"
    project = tmp_path / "project"
    project.mkdir()
    monkeypatch.chdir(project)
    with patch("graphify.__main__.Path.home", return_value=home):
        monkeypatch.setattr(sys, "argv", ["graphify", "amp", "install"])
        main()
        skill_dir = home / ".config" / "agents" / "skills" / "graphify"
        assert (skill_dir / "SKILL.md").exists()
        # A representative reference from the shipped amp bundle lands too.
        assert (skill_dir / "references" / "exports.md").exists()
        assert (skill_dir / "references" / "hooks.md").exists()

        monkeypatch.setattr(sys, "argv", ["graphify", "amp", "uninstall"])
        main()

    assert not skill_dir.exists()
