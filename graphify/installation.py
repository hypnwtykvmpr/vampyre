"""Canonical installation guidance for Vampyre."""

from __future__ import annotations

from collections.abc import Iterable


VAMPYRE_UV_SOURCE = "git+https://github.com/hypnwtykvmpr/vampyre.git@v0.9.5"


def uv_tool_install_command(
    extra: str | None = None,
    *,
    with_packages: Iterable[str] = (),
) -> str:
    """Return a uv command that installs Vampyre and optional dependencies."""
    package = f"graphifyy[{extra}]" if extra else "graphifyy"
    additions = "".join(f' --with "{dependency}"' for dependency in with_packages)
    return f'uv tool install --force{additions} "{package} @ {VAMPYRE_UV_SOURCE}"'
