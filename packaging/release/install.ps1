$ErrorActionPreference = "Stop"

$wheel = Get-ChildItem -LiteralPath $PSScriptRoot -Filter "graphifyy-*.whl" -File |
    Select-Object -First 1
if (-not $wheel) {
    throw "Vampyre wheel not found beside this installer."
}
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    throw "uv is required: https://docs.astral.sh/uv/getting-started/installation/"
}

& uv tool install --force $wheel.FullName
if ($LASTEXITCODE -ne 0) {
    throw "uv failed to install Vampyre (exit $LASTEXITCODE)."
}
$toolBin = (& uv tool dir --bin).Trim()
if ($LASTEXITCODE -ne 0) {
    throw "uv could not locate its tool binary directory (exit $LASTEXITCODE)."
}
$graphify = Join-Path $toolBin "graphify.exe"
& $graphify --version
if ($LASTEXITCODE -ne 0) {
    throw "Vampyre installed but graphify --version failed (exit $LASTEXITCODE)."
}
