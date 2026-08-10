```powershell
# Detect Python in Vampyre's uv tool environment.
New-Item -ItemType Directory -Force -Path graphify-out | Out-Null
$GRAPHIFY_PYTHON = $null

function Find-GraphifyPython {
    # `uv tool dir` is authoritative and respects UV_TOOL_DIR automatically.
    if (Get-Command uv -ErrorAction SilentlyContinue) {
        $uvDir = (uv tool dir 2>$null).Trim()
        if ($uvDir) {
            $py = Join-Path $uvDir "graphifyy\Scripts\python.exe"
            if (Test-Path $py) {
                & $py -c "import graphify" 2>$null
                if ($LASTEXITCODE -eq 0) { return $py }
            }
        }
    }
    return $null
}

# Try to find the installed Vampyre tool first.
$GRAPHIFY_PYTHON = Find-GraphifyPython

# Not found — install then re-detect
if (-not $GRAPHIFY_PYTHON) {
    if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
        throw "graphify requires uv; install uv first, then retry."
    }
    uv tool install --force "graphifyy @ git+https://github.com/hypnwtykvmpr/vampyre.git@v0.9.5"
    $GRAPHIFY_PYTHON = Find-GraphifyPython
}
if (-not $GRAPHIFY_PYTHON) { throw "graphify's uv tool interpreter could not be resolved." }

# Save interpreter path — all subsequent steps read this
$GRAPHIFY_PYTHON | Out-File -FilePath graphify-out\.graphify_python -Encoding utf8 -NoNewline
```

If the import succeeds, print nothing and move straight to Step 2.

**In every subsequent block, run Python through the saved interpreter — `& (Get-Content graphify-out\.graphify_python)` in place of a bare `python3` — so every step uses the interpreter that actually has graphify.**
