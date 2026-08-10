#!/usr/bin/env sh
set -eu

bundle_dir=$(CDPATH= cd "$(dirname "$0")" && pwd)
wheel=
for candidate in "$bundle_dir"/graphifyy-*.whl; do
    if [ -f "$candidate" ]; then
        wheel=$candidate
        break
    fi
done

if [ -z "$wheel" ]; then
    printf '%s\n' "Vampyre wheel not found beside this installer." >&2
    exit 1
fi
if ! command -v uv >/dev/null 2>&1; then
    printf '%s\n' "uv is required: https://docs.astral.sh/uv/getting-started/installation/" >&2
    exit 1
fi

uv tool install --force "$wheel"
tool_bin=$(uv tool dir --bin)
"$tool_bin/graphify" --version
