"""Fork-specific installation guidance."""

from graphify.installation import FORK_UV_SOURCE, uv_tool_install_command


def test_default_install_command_targets_v9_fork() -> None:
    assert uv_tool_install_command() == (
        'uv tool install --force "graphifyy @ git+https://github.com/hypnwtykvmpr/vampyre.git@v9"'
    )
    assert FORK_UV_SOURCE in uv_tool_install_command()


def test_extra_install_command_preserves_fork_source() -> None:
    assert uv_tool_install_command("mcp") == (
        'uv tool install --force "graphifyy[mcp] @ '
        'git+https://github.com/hypnwtykvmpr/vampyre.git@v9"'
    )


def test_additional_package_install_uses_uv_tool_environment() -> None:
    assert uv_tool_install_command(with_packages=("tomli",)) == (
        'uv tool install --force --with "tomli" "graphifyy @ '
        'git+https://github.com/hypnwtykvmpr/vampyre.git@v9"'
    )
