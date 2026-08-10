"""Build and validate deterministic Vampyre release bundles."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import os
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from collections.abc import Callable, Iterable
from pathlib import Path, PurePosixPath

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib  # type: ignore[no-redef]


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SUPPORTED_PLATFORMS = {"linux", "macos", "windows"}


def _project_version(pyproject: Path) -> str:
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    return str(data["project"]["version"])


def verify_release_tag(pyproject: Path, tag: str) -> str:
    """Return the project version when ``tag`` is its exact ``v``-prefixed form."""
    version = _project_version(pyproject)
    if tag != f"v{version}":
        raise ValueError(f"release tag {tag!r} does not match project version {version!r}")
    return version


def _archive_entries(
    *, platform: str, version: str, wheel: Path, project_root: Path
) -> dict[str, tuple[bytes, int]]:
    if platform not in SUPPORTED_PLATFORMS:
        raise ValueError(f"unsupported platform: {platform}")
    if not wheel.is_file() or not wheel.name.startswith(f"graphifyy-{version}-"):
        raise ValueError(f"wheel does not match release version {version}: {wheel}")

    bundle_root = f"vampyre-{version}-{platform}"
    release_root = project_root / "packaging" / "release"
    installer = "install.ps1" if platform == "windows" else "install.sh"
    sources = {
        wheel.name: wheel,
        "INSTALL.md": release_root / "INSTALL.md",
        "LICENSE": project_root / "LICENSE",
        installer: release_root / installer,
    }
    entries: dict[str, tuple[bytes, int]] = {}
    for name, source in sources.items():
        if not source.is_file():
            raise FileNotFoundError(source)
        mode = 0o755 if name.startswith("install.") else 0o644
        entries[f"{bundle_root}/{name}"] = (source.read_bytes(), mode)
    return entries


def _write_zip(output: Path, entries: dict[str, tuple[bytes, int]]) -> None:
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name in sorted(entries):
            payload, mode = entries[name]
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = mode << 16
            archive.writestr(info, payload)


def _write_tar_gz(output: Path, entries: dict[str, tuple[bytes, int]]) -> None:
    with output.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with tarfile.open(
                fileobj=compressed,
                mode="w",
                format=tarfile.PAX_FORMAT,
                encoding="utf-8",
                errors="strict",
            ) as archive:
                for name in sorted(entries):
                    payload, mode = entries[name]
                    info = tarfile.TarInfo(name)
                    info.size = len(payload)
                    info.mode = mode
                    info.mtime = 0
                    info.uid = 0
                    info.gid = 0
                    info.uname = ""
                    info.gname = ""
                    archive.addfile(info, io.BytesIO(payload))


def _replace_archive(path: Path, writer: Callable[[Path], None]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as temporary:
        temporary_path = Path(temporary.name)
    try:
        writer(temporary_path)
        temporary_path.chmod(0o644)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _normalize_wheel(wheel: Path) -> None:
    with zipfile.ZipFile(wheel) as archive:
        infos = archive.infolist()
        if any(info.is_dir() for info in infos):
            raise ValueError(f"wheel contains a directory entry: {wheel}")
        if len({info.filename for info in infos}) != len(infos):
            raise ValueError(f"wheel contains duplicate members: {wheel}")
        entries = {
            info.filename: (
                archive.read(info),
                0o755 if ((info.external_attr >> 16) & 0o111) else 0o644,
            )
            for info in infos
        }

    _replace_archive(wheel, lambda output: _write_zip(output, entries))


def _normalize_sdist(sdist: Path, *, version: str) -> None:
    root = f"graphifyy-{version}"
    entries: list[tuple[str, bytes | None, int]] = []
    with tarfile.open(sdist, "r:gz", encoding="utf-8", errors="strict") as archive:
        members = archive.getmembers()
        if len({member.name for member in members}) != len(members):
            raise ValueError(f"source distribution contains duplicate members: {sdist}")
        for member in members:
            if member.name != root and not member.name.startswith(f"{root}/"):
                raise ValueError(f"source distribution member is outside {root!r}: {member.name!r}")
            if member.isdir():
                entries.append((member.name, None, 0o755))
                continue
            if not member.isfile():
                raise ValueError(f"unsupported source distribution member: {member.name!r}")
            source = archive.extractfile(member)
            if source is None:
                raise ValueError(f"unreadable source distribution member: {member.name!r}")
            mode = 0o755 if member.mode & 0o111 else 0o644
            entries.append((member.name, source.read(), mode))

    def _write(output: Path) -> None:
        with output.open("wb") as raw:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
                with tarfile.open(
                    fileobj=compressed,
                    mode="w",
                    format=tarfile.PAX_FORMAT,
                    encoding="utf-8",
                    errors="strict",
                ) as archive:
                    for name, payload, mode in sorted(entries):
                        info = tarfile.TarInfo(name)
                        info.mode = mode
                        info.mtime = 0
                        info.uid = 0
                        info.gid = 0
                        info.uname = ""
                        info.gname = ""
                        if payload is None:
                            info.type = tarfile.DIRTYPE
                            archive.addfile(info)
                        else:
                            info.size = len(payload)
                            archive.addfile(info, io.BytesIO(payload))

    _replace_archive(sdist, _write)


def normalize_distributions(*, dist_dir: Path, version: str) -> tuple[Path, Path]:
    """Normalize release distributions to reproducible, private-free archives."""
    wheel = _find_wheel(dist_dir, version)
    sdists = sorted(dist_dir.glob(f"graphifyy-{version}.tar.gz"))
    if len(sdists) != 1:
        raise ValueError(
            f"expected one graphifyy {version} source distribution in {dist_dir}, "
            f"found {len(sdists)}"
        )
    sdist = sdists[0]
    _normalize_wheel(wheel)
    _normalize_sdist(sdist, version=version)
    return wheel, sdist


def build_bundle(
    *, platform: str, version: str, wheel: Path, output_dir: Path, project_root: Path
) -> Path:
    """Create one deterministic platform bundle around the universal wheel."""
    entries = _archive_entries(
        platform=platform,
        version=version,
        wheel=wheel,
        project_root=project_root,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = ".zip" if platform == "windows" else ".tar.gz"
    output = output_dir / f"vampyre-{version}-{platform}{suffix}"
    if platform == "windows":
        _write_zip(output, entries)
    else:
        _write_tar_gz(output, entries)
    return output


def build_graph_bundle(*, version: str, graph_dir: Path, output_dir: Path) -> Path:
    """Package the self-graph using the same deterministic tar writer."""
    required = [graph_dir / "graph.json", graph_dir / "graph.html"]
    optional = [graph_dir / "GRAPH_REPORT.md"]
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(", ".join(str(path) for path in missing))
    entries = {
        f"vampyre-{version}-self-graph/{path.name}": (path.read_bytes(), 0o644)
        for path in required + optional
        if path.is_file()
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"vampyre-{version}-self-graph.tar.gz"
    _write_tar_gz(output, entries)
    return output


def write_checksums(assets: Iterable[Path], output: Path) -> Path:
    """Write a stable SHA-256 manifest for release assets."""
    rows = []
    for asset in sorted((Path(path) for path in assets), key=lambda path: path.name):
        if asset.resolve() == output.resolve():
            continue
        digest = hashlib.sha256(asset.read_bytes()).hexdigest()
        rows.append(f"{digest}  {asset.name}")
    if not rows:
        raise ValueError("no release assets supplied for checksumming")
    output.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return output


def _find_wheel(dist_dir: Path, version: str) -> Path:
    wheels = sorted(dist_dir.glob(f"graphifyy-{version}-*.whl"))
    if len(wheels) != 1:
        raise ValueError(
            f"expected one graphifyy {version} wheel in {dist_dir}, found {len(wheels)}"
        )
    return wheels[0]


def _safe_bundle_member_target(name: str, destination: Path) -> Path:
    """Resolve one POSIX archive member beneath ``destination``."""
    archive_path = PurePosixPath(name)
    parts = archive_path.parts
    if not parts or archive_path.is_absolute() or ".." in parts:
        raise ValueError(f"unsafe release archive member: {name!r}")
    target = destination.joinpath(*parts)
    try:
        target.resolve().relative_to(destination.resolve())
    except ValueError as exc:
        raise ValueError(f"unsafe release archive member: {name!r}") from exc
    return target


def _safe_extract_bundle(bundle: Path, destination: Path) -> None:
    """Extract a release bundle while refusing absolute or parent paths."""
    destination.mkdir(parents=True, exist_ok=True)

    if bundle.suffix == ".zip":
        with zipfile.ZipFile(bundle) as archive:
            for info in archive.infolist():
                target = _safe_bundle_member_target(info.filename, destination)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(archive.read(info))
                target.chmod((info.external_attr >> 16) & 0o777 or 0o644)
        return

    with tarfile.open(bundle, "r:gz", encoding="utf-8", errors="strict") as archive:
        for member in archive.getmembers():
            if not member.isfile():
                raise ValueError(f"unsupported release archive member: {member.name!r}")
            source = archive.extractfile(member)
            if source is None:
                raise ValueError(f"unreadable release archive member: {member.name!r}")
            target = _safe_bundle_member_target(member.name, destination)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(source.read())
            target.chmod(member.mode & 0o777)


def smoke_release(
    *, platform: str, version: str, dist_dir: Path, output_dir: Path, project_root: Path
) -> Path:
    """Run the packaged installer in an isolated uv tool root and execute the CLI."""
    wheel = _find_wheel(dist_dir, version)
    bundle = build_bundle(
        platform=platform,
        version=version,
        wheel=wheel,
        output_dir=output_dir,
        project_root=project_root,
    )
    with tempfile.TemporaryDirectory(prefix="vampyre-release-smoke-") as temporary:
        root = Path(temporary)
        bin_dir = root / "bin"
        env = os.environ.copy()
        env["UV_TOOL_DIR"] = str(root / "tools")
        env["UV_TOOL_BIN_DIR"] = str(bin_dir)
        env["PATH"] = os.pathsep.join([str(bin_dir), env.get("PATH", "")])
        _safe_extract_bundle(bundle, root / "unpacked")
        bundle_root = root / "unpacked" / f"vampyre-{version}-{platform}"
        if platform == "windows":
            installer_command = [
                "pwsh",
                "-NoProfile",
                "-File",
                str(bundle_root / "install.ps1"),
            ]
        else:
            installer_command = ["sh", str(bundle_root / "install.sh")]
        subprocess.run(installer_command, check=True, env=env)
        executable = bin_dir / ("graphify.exe" if platform == "windows" else "graphify")
        version_result = subprocess.run(
            [str(executable), "--version"],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            env=env,
        )
        if version_result.stdout.strip() != f"graphify {version}":
            raise RuntimeError(f"unexpected CLI version: {version_result.stdout!r}")
        subprocess.run(
            [str(executable), "--help"],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            env=env,
        )
    return bundle


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    verify = subparsers.add_parser("verify-tag")
    verify.add_argument("--tag", required=True)
    verify.add_argument("--pyproject", type=Path, default=PROJECT_ROOT / "pyproject.toml")

    bundle = subparsers.add_parser("bundle")
    bundle.add_argument("--platform", choices=sorted(SUPPORTED_PLATFORMS), required=True)
    bundle.add_argument("--version", required=True)
    bundle.add_argument("--wheel", type=Path, required=True)
    bundle.add_argument("--output-dir", type=Path, required=True)

    smoke = subparsers.add_parser("smoke")
    smoke.add_argument("--platform", choices=sorted(SUPPORTED_PLATFORMS), required=True)
    smoke.add_argument("--version", required=True)
    smoke.add_argument("--dist-dir", type=Path, required=True)
    smoke.add_argument("--output-dir", type=Path, required=True)

    graph = subparsers.add_parser("graph-bundle")
    graph.add_argument("--version", required=True)
    graph.add_argument("--graph-dir", type=Path, required=True)
    graph.add_argument("--output-dir", type=Path, required=True)

    normalize = subparsers.add_parser("normalize-dist")
    normalize.add_argument("--version", required=True)
    normalize.add_argument("--dist-dir", type=Path, required=True)

    checksums = subparsers.add_parser("checksums")
    checksums.add_argument("--output", type=Path, required=True)
    checksums.add_argument("assets", nargs="+", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "verify-tag":
        print(verify_release_tag(args.pyproject, args.tag))
    elif args.command == "bundle":
        print(
            build_bundle(
                platform=args.platform,
                version=args.version,
                wheel=args.wheel,
                output_dir=args.output_dir,
                project_root=PROJECT_ROOT,
            )
        )
    elif args.command == "smoke":
        print(
            smoke_release(
                platform=args.platform,
                version=args.version,
                dist_dir=args.dist_dir,
                output_dir=args.output_dir,
                project_root=PROJECT_ROOT,
            )
        )
    elif args.command == "graph-bundle":
        print(
            build_graph_bundle(
                version=args.version,
                graph_dir=args.graph_dir,
                output_dir=args.output_dir,
            )
        )
    elif args.command == "normalize-dist":
        for path in normalize_distributions(
            dist_dir=args.dist_dir,
            version=args.version,
        ):
            print(path)
    elif args.command == "checksums":
        print(write_checksums(args.assets, args.output))
    return 0


if __name__ == "__main__":
    sys.exit(main())
