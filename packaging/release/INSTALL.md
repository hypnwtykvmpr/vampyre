# Install Vampyre

This bundle contains the universal Vampyre Python wheel that was installed and
smoke-tested on the operating system named in the archive filename.

Requirements:

- Python 3.10 or newer
- [uv](https://docs.astral.sh/uv/getting-started/installation/)

On Linux or macOS:

```sh
./install.sh
```

On Windows PowerShell:

```powershell
.\install.ps1
```

The installer uses `uv tool install --force` with the wheel beside it, then
runs `graphify --version`. Optional features can be installed from the exact
release tag; see the repository README for the available extras.

The canonical source, documentation, security policy, and release checksums are
maintained at https://github.com/hypnwtykvmpr/vampyre. No parent or predecessor
repository is a Vampyre update, support, or release channel.
