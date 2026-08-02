"""Canonical installation guidance for the Vampyre fork."""

from __future__ import annotations

from collections.abc import Iterable


FORK_UV_SOURCE = "git+https://github.com/hypnwtykvmpr/vampyre.git@v9"


def uv_tool_install_command(
    extra: str | None = None,
    *,
    with_packages: Iterable[str] = (),
) -> str:
    """Return a uv command that installs this fork and optional dependencies."""
    package = f"graphifyy[{extra}]" if extra else "graphifyy"
    additions = "".join(f' --with "{dependency}"' for dependency in with_packages)
    return f'uv tool install --force{additions} "{package} @ {FORK_UV_SOURCE}"'
