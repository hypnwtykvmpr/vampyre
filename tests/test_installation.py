"""Canonical Vampyre installation guidance."""

from graphify.installation import VAMPYRE_UV_SOURCE, uv_tool_install_command


def test_default_install_command_targets_current_release() -> None:
    assert uv_tool_install_command() == (
        'uv tool install --force "graphifyy @ '
        'git+https://github.com/hypnwtykvmpr/vampyre.git@v0.9.5"'
    )
    assert VAMPYRE_UV_SOURCE in uv_tool_install_command()


def test_extra_install_command_preserves_release_source() -> None:
    assert uv_tool_install_command("mcp") == (
        'uv tool install --force "graphifyy[mcp] @ '
        'git+https://github.com/hypnwtykvmpr/vampyre.git@v0.9.5"'
    )


def test_additional_package_install_uses_uv_tool_environment() -> None:
    assert uv_tool_install_command(with_packages=("tomli",)) == (
        'uv tool install --force --with "tomli" "graphifyy @ '
        'git+https://github.com/hypnwtykvmpr/vampyre.git@v0.9.5"'
    )
