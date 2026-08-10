from __future__ import annotations

import gzip
import hashlib
import io
import tarfile
import zipfile
from pathlib import Path

import pytest
import tools.release_bundle as release_bundle

from tools.release_bundle import (
    _safe_bundle_member_target,
    _safe_extract_bundle,
    build_bundle,
    verify_release_tag,
    write_checksums,
)


ROOT = Path(__file__).resolve().parents[1]
VERSION = "0.9.5"


@pytest.fixture
def fake_wheel(tmp_path: Path) -> Path:
    wheel = tmp_path / f"graphifyy-{VERSION}-py3-none-any.whl"
    wheel.write_bytes(b"portable wheel payload")
    return wheel


def _write_test_archive(path: Path, members: dict[str, bytes]) -> None:
    if path.suffix == ".zip":
        with zipfile.ZipFile(path, "w") as archive:
            for name, payload in members.items():
                archive.writestr(name, payload)
        return

    with tarfile.open(path, "w:gz", encoding="utf-8", errors="strict") as archive:
        for name, payload in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))


def _write_distribution_pair(
    directory: Path,
    *,
    timestamp: int,
    owner: str,
    reverse: bool,
) -> None:
    directory.mkdir()
    wheel = directory / f"graphifyy-{VERSION}-py3-none-any.whl"
    wheel_entries = [
        ("graphify/__init__.py", b'__version__ = "0.9.5"\n'),
        ("graphifyy-0.9.5.dist-info/METADATA", b"Version: 0.9.5\n"),
    ]
    if reverse:
        wheel_entries.reverse()
    with zipfile.ZipFile(wheel, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, payload in wheel_entries:
            info = zipfile.ZipInfo(name, date_time=(2024, 1, 2, 3, 4, 6))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o644 << 16
            archive.writestr(info, payload)

    sdist = directory / f"graphifyy-{VERSION}.tar.gz"
    sdist_entries = [
        (f"graphifyy-{VERSION}", None, 0o755),
        (f"graphifyy-{VERSION}/PKG-INFO", b"Version: 0.9.5\n", 0o644),
        (f"graphifyy-{VERSION}/graphify", None, 0o755),
        (
            f"graphifyy-{VERSION}/graphify/__init__.py",
            b'__version__ = "0.9.5"\n',
            0o644,
        ),
    ]
    if reverse:
        sdist_entries.reverse()
    with sdist.open("wb") as raw:
        with gzip.GzipFile(filename=owner, mode="wb", fileobj=raw, mtime=timestamp) as zipped:
            with tarfile.open(
                fileobj=zipped,
                mode="w",
                format=tarfile.PAX_FORMAT,
                encoding="utf-8",
                errors="strict",
            ) as archive:
                for name, payload, mode in sdist_entries:
                    info = tarfile.TarInfo(name)
                    info.mode = mode
                    info.mtime = timestamp
                    info.uid = timestamp
                    info.gid = timestamp + 1
                    info.uname = owner
                    info.gname = f"{owner}-group"
                    if payload is None:
                        info.type = tarfile.DIRTYPE
                        archive.addfile(info)
                    else:
                        info.size = len(payload)
                        archive.addfile(info, io.BytesIO(payload))


@pytest.mark.parametrize(
    ("platform", "suffix", "installer"),
    [
        ("windows", ".zip", "install.ps1"),
        ("macos", ".tar.gz", "install.sh"),
        ("linux", ".tar.gz", "install.sh"),
    ],
)
def test_platform_bundle_contains_wheel_license_notes_and_installer(
    tmp_path: Path,
    fake_wheel: Path,
    platform: str,
    suffix: str,
    installer: str,
) -> None:
    output = build_bundle(
        platform=platform,
        version=VERSION,
        wheel=fake_wheel,
        output_dir=tmp_path / "assets",
        project_root=ROOT,
    )

    assert output.name == f"vampyre-{VERSION}-{platform}{suffix}"
    if platform == "windows":
        with zipfile.ZipFile(output) as archive:
            members = set(archive.namelist())
    else:
        with tarfile.open(output, "r:gz", encoding="utf-8", errors="strict") as archive:
            members = set(archive.getnames())
    root = f"vampyre-{VERSION}-{platform}"
    assert members == {
        f"{root}/{fake_wheel.name}",
        f"{root}/INSTALL.md",
        f"{root}/LICENSE",
        f"{root}/{installer}",
    }
    first_digest = hashlib.sha256(output.read_bytes()).digest()
    rebuilt = build_bundle(
        platform=platform,
        version=VERSION,
        wheel=fake_wheel,
        output_dir=tmp_path / "assets",
        project_root=ROOT,
    )
    assert hashlib.sha256(rebuilt.read_bytes()).digest() == first_digest


def test_posix_installer_uses_shell_glob_for_sibling_wheel() -> None:
    installer = (ROOT / "packaging" / "release" / "install.sh").read_text(encoding="utf-8")

    assert "find " not in installer
    assert '"$bundle_dir"/graphifyy-*.whl' in installer


def test_posix_installer_avoids_non_posix_double_dash_operands() -> None:
    installer = (ROOT / "packaging" / "release" / "install.sh").read_text(encoding="utf-8")

    assert "cd --" not in installer
    assert "dirname --" not in installer


@pytest.mark.parametrize("suffix", [".zip", ".tar.gz"])
@pytest.mark.parametrize(
    "unsafe_member",
    ["../escape.txt", "nested/../../escape.txt", "/absolute.txt"],
)
def test_release_bundle_extraction_refuses_unsafe_members(
    tmp_path: Path,
    suffix: str,
    unsafe_member: str,
) -> None:
    bundle = tmp_path / f"malicious{suffix}"
    _write_test_archive(bundle, {unsafe_member: b"must not escape"})

    with pytest.raises(ValueError, match="unsafe release archive member"):
        _safe_extract_bundle(bundle, tmp_path / "unpacked")


@pytest.mark.parametrize("suffix", [".zip", ".tar.gz"])
def test_release_bundle_extraction_accepts_nested_files(
    tmp_path: Path,
    suffix: str,
) -> None:
    bundle = tmp_path / f"benign{suffix}"
    members = {"ok.txt": b"root", "sub/deep.txt": b"nested"}
    _write_test_archive(bundle, members)

    destination = tmp_path / "unpacked"
    _safe_extract_bundle(bundle, destination)

    assert (destination / "ok.txt").read_bytes() == b"root"
    assert (destination / "sub" / "deep.txt").read_bytes() == b"nested"


@pytest.mark.parametrize("name", ["", ".", "/absolute.txt", "../escape.txt"])
def test_release_member_target_rejects_invalid_posix_names(
    tmp_path: Path,
    name: str,
) -> None:
    with pytest.raises(ValueError, match="unsafe release archive member"):
        _safe_bundle_member_target(name, tmp_path / "unpacked")


def test_release_member_target_enforces_resolved_containment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "unpacked"
    target = destination / "safe.txt"
    escaped = tmp_path / "outside" / "safe.txt"
    original_resolve = Path.resolve

    def _resolve(path: Path, strict: bool = False) -> Path:
        if path == target:
            return escaped
        return original_resolve(path, strict=strict)

    monkeypatch.setattr(Path, "resolve", _resolve)

    with pytest.raises(ValueError, match="unsafe release archive member"):
        _safe_bundle_member_target("safe.txt", destination)


def test_release_tag_must_exactly_match_project_version() -> None:
    assert verify_release_tag(ROOT / "pyproject.toml", "v0.9.5") == VERSION
    with pytest.raises(ValueError, match="does not match project version"):
        verify_release_tag(ROOT / "pyproject.toml", "v0.9.4")


def test_checksums_are_sorted_and_cover_every_asset(tmp_path: Path) -> None:
    beta = tmp_path / "beta.zip"
    alpha = tmp_path / "alpha.tar.gz"
    beta.write_bytes(b"beta")
    alpha.write_bytes(b"alpha")

    checksum_path = write_checksums([beta, alpha], tmp_path / "SHA256SUMS")

    assert checksum_path.read_text(encoding="utf-8").splitlines() == [
        f"{hashlib.sha256(b'alpha').hexdigest()}  alpha.tar.gz",
        f"{hashlib.sha256(b'beta').hexdigest()}  beta.zip",
    ]


def test_distribution_normalization_is_reproducible_and_private(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    _write_distribution_pair(first, timestamp=1_700_000_000, owner="alice", reverse=False)
    _write_distribution_pair(second, timestamp=1_800_000_000, owner="bob", reverse=True)

    first_outputs = release_bundle.normalize_distributions(
        dist_dir=first,
        version=VERSION,
    )
    second_outputs = release_bundle.normalize_distributions(
        dist_dir=second,
        version=VERSION,
    )

    assert [path.name for path in first_outputs] == [path.name for path in second_outputs]
    assert [path.read_bytes() for path in first_outputs] == [
        path.read_bytes() for path in second_outputs
    ]

    wheel, sdist = first_outputs
    with zipfile.ZipFile(wheel) as archive:
        assert archive.namelist() == sorted(archive.namelist())
        assert {info.date_time for info in archive.infolist()} == {(1980, 1, 1, 0, 0, 0)}
        assert archive.read("graphify/__init__.py") == b'__version__ = "0.9.5"\n'
    with tarfile.open(sdist, "r:gz", encoding="utf-8", errors="strict") as archive:
        members = archive.getmembers()
        assert [member.name for member in members] == sorted(member.name for member in members)
        assert {member.mtime for member in members} == {0}
        assert {member.uid for member in members} == {0}
        assert {member.gid for member in members} == {0}
        assert {member.uname for member in members} == {""}
        assert {member.gname for member in members} == {""}
        source = archive.extractfile(f"graphifyy-{VERSION}/graphify/__init__.py")
        assert source is not None
        assert source.read() == b'__version__ = "0.9.5"\n'


def test_release_workflow_builds_and_smoke_tests_every_supported_os() -> None:
    workflow = (ROOT / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")

    assert "tags:" in workflow
    assert '"v*"' in workflow
    assert "uv build" in workflow
    assert "release_bundle normalize-dist" in workflow
    assert "ubuntu-latest" in workflow
    assert "macos-latest" in workflow
    assert "windows-latest" in workflow
    assert "release_bundle smoke" in workflow
    assert "softprops/action-gh-release@v2" in workflow
    assert "fail_on_unmatched_files: true" in workflow
    assert "SHA256SUMS" in workflow
    assert "release_bundle graph-bundle" in workflow
    assert "permissions:\n  contents: read" in workflow
    assert workflow.count("timeout-minutes: 30") == 4
    assert workflow.count('save-cache: "false"') == 4
    assert (
        "body_path: docs/releases/${{ needs.build-distributions.outputs.version }}.md" in workflow
    )
    assert "cancel-in-progress: false" in workflow
    assert 'test "$GITHUB_SHA" = "$(git rev-parse origin/main)"' in workflow
