from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest


_PROJECT_ROOT = Path(__file__).parents[1]
_POLICY = (_PROJECT_ROOT / "conftest.py").read_text(encoding="utf-8")
_SERIAL: tuple[str, ...] = ()
_XDIST = ("-n", "2", "--dist", "loadgroup")


def _run_isolated_pytest(
    tmp_path: Path,
    source: str,
    parallel_args: tuple[str, ...],
    *,
    extra_plugin: str | None = None,
    selection_args: tuple[str, ...] = (),
) -> subprocess.CompletedProcess[str]:
    (tmp_path / "conftest.py").write_text(_POLICY, encoding="utf-8")
    (tmp_path / "test_policy_case.py").write_text(source, encoding="utf-8")
    if extra_plugin is not None:
        (tmp_path / "selection_plugin.py").write_text(extra_plugin, encoding="utf-8")

    env = os.environ.copy()
    env.pop("PYTEST_ADDOPTS", None)
    command = [sys.executable, "-m", "pytest", "test_policy_case.py", "-q"]
    command.extend(parallel_args)
    command.extend(selection_args)
    if extra_plugin is not None:
        command.extend(("-p", "selection_plugin"))
    return subprocess.run(
        command,
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )


@pytest.mark.parametrize("parallel_args", [_SERIAL, _XDIST], ids=["serial", "xdist"])
@pytest.mark.parametrize(
    "source, expected",
    [
        ("import pytest\n@pytest.mark.xfail\ndef test_case():\n    assert False\n", "xfail"),
        ("import pytest\n@pytest.mark.xfail\ndef test_case():\n    pass\n", "xpass"),
    ],
)
def test_xfail_outcomes_fail_the_process(tmp_path, parallel_args, source, expected):
    result = _run_isolated_pytest(tmp_path, source, parallel_args)

    assert result.returncode != 0, result.stdout + result.stderr
    assert f"produced {expected}" in result.stdout + result.stderr


@pytest.mark.parametrize("parallel_args", [_SERIAL, _XDIST], ids=["serial", "xdist"])
@pytest.mark.parametrize(
    "selection_args",
    [
        ("-k", "selected"),
        ("-m", "focus"),
        ("--deselect", "test_policy_case.py::test_other"),
    ],
    ids=["keyword", "marker", "explicit-node"],
)
def test_cli_deselection_fails_the_process(tmp_path, parallel_args, selection_args):
    source = (
        "import pytest\n"
        "@pytest.mark.focus\n"
        "def test_selected():\n"
        "    pass\n\n"
        "def test_other():\n"
        "    pass\n"
    )
    result = _run_isolated_pytest(
        tmp_path,
        source,
        parallel_args,
        selection_args=selection_args,
    )

    assert result.returncode != 0, result.stdout + result.stderr
    assert "POLICY VIOLATION" in result.stdout + result.stderr
    assert "deselect" in (result.stdout + result.stderr).lower()


@pytest.mark.parametrize("parallel_args", [_SERIAL, _XDIST], ids=["serial", "xdist"])
def test_plugin_deselection_fails_the_process(tmp_path, parallel_args):
    source = "def test_selected():\n    pass\n\ndef test_other():\n    pass\n"
    plugin = """
def pytest_collection_modifyitems(config, items):
    deselected = [items.pop()]
    config.hook.pytest_deselected(items=deselected)
"""
    result = _run_isolated_pytest(tmp_path, source, parallel_args, extra_plugin=plugin)

    assert result.returncode != 0, result.stdout + result.stderr
    assert "POLICY VIOLATION" in result.stdout + result.stderr
    assert "deselect" in (result.stdout + result.stderr).lower()


def test_worker_crash_preserves_xdist_diagnostics(tmp_path):
    source = "import os\n\ndef test_crashes():\n    os._exit(1)\n\ndef test_control():\n    pass\n"
    result = _run_isolated_pytest(tmp_path, source, _XDIST)
    output = result.stdout + result.stderr

    assert result.returncode != 0, output
    assert "INTERNALERROR" not in output
    assert "AttributeError" not in output
    assert "test_policy_case.py::test_crashes" in output


@pytest.mark.parametrize("parallel_args", [_SERIAL, _XDIST], ids=["serial", "xdist"])
def test_plain_pass_remains_successful(tmp_path, parallel_args):
    result = _run_isolated_pytest(tmp_path, "def test_case():\n    pass\n", parallel_args)

    assert result.returncode == 0, result.stdout + result.stderr
    assert "1 passed" in result.stdout
