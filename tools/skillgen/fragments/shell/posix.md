```bash
# Detect the exact Python interpreter owned by Vampyre's uv tool environment.
PYTHON=""
GRAPHIFY_BIN=$(command -v graphify 2>/dev/null)
if [ -z "$GRAPHIFY_BIN" ]; then
    if ! command -v uv >/dev/null 2>&1; then
        echo "graphify requires uv; install uv first, then retry." >&2
        exit 1
    fi
    uv tool install --force "graphifyy @ git+https://github.com/hypnwtykvmpr/vampyre.git@v0.9.5"
    GRAPHIFY_BIN=$(command -v graphify 2>/dev/null)
    if [ -z "$GRAPHIFY_BIN" ]; then
        GRAPHIFY_BIN="$(uv tool dir --bin)/graphify"
    fi
fi
if [ -f "$GRAPHIFY_BIN" ]; then
    _SHEBANG=$(head -1 "$GRAPHIFY_BIN" | tr -d '#!')
    case "$_SHEBANG" in
        *[!a-zA-Z0-9/_.@-]*) ;;
        *) "$_SHEBANG" -c "import graphify" 2>/dev/null && PYTHON="$_SHEBANG" ;;
    esac
fi
if [ -z "$PYTHON" ]; then
    echo "graphify's uv tool interpreter could not be resolved." >&2
    exit 1
fi
# Write interpreter path for all subsequent steps (persists across invocations)
mkdir -p graphify-out
"$PYTHON" -c "import sys; open('graphify-out/.graphify_python', 'w', encoding='utf-8').write(sys.executable)"
```

If the import succeeds, print nothing and move straight to Step 2.

**In every subsequent bash block, replace `python3` with `"$(cat graphify-out/.graphify_python)"` to use the correct interpreter.**
