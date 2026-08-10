Claude Code on Windows runs skill commands in Git Bash. Run every command block in this skill in Git Bash, not PowerShell or `cmd.exe`.

```bash
# Resolve the exact Python interpreter owned by Vampyre's uv tool environment.
if ! command -v uv >/dev/null 2>&1; then
    echo "graphify requires uv; install uv first, then retry." >&2
    exit 1
fi

find_graphify_python() {
    local uv_dir candidate
    uv_dir=$(uv tool dir 2>/dev/null) || return 1
    if command -v cygpath >/dev/null 2>&1; then
        uv_dir=$(cygpath -u "$uv_dir")
    fi
    candidate="$uv_dir/graphifyy/Scripts/python.exe"
    if [ -f "$candidate" ] && "$candidate" -c "import graphify" 2>/dev/null; then
        printf '%s\n' "$candidate"
        return 0
    fi
    return 1
}

PYTHON=$(find_graphify_python || true)
if [ -z "$PYTHON" ]; then
    uv tool install --force "graphifyy @ git+https://github.com/hypnwtykvmpr/vampyre.git@v0.9.5"
    PYTHON=$(find_graphify_python || true)
fi
if [ -z "$PYTHON" ]; then
    echo "graphify's uv tool interpreter could not be resolved." >&2
    exit 1
fi

mkdir -p graphify-out
"$PYTHON" -c "import sys; open('graphify-out/.graphify_python', 'w', encoding='utf-8').write(sys.executable)"
```

If the import succeeds, print nothing and move straight to Step 2.

**In every subsequent bash block, invoke `"$(cat graphify-out/.graphify_python)"` so paths containing spaces remain one executable path.**
