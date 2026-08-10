# Vampyre

[![CI](https://github.com/hypnwtykvmpr/vampyre/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/hypnwtykvmpr/vampyre/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/release/hypnwtykvmpr/vampyre)](https://github.com/hypnwtykvmpr/vampyre/releases)
[![Python](https://img.shields.io/badge/python-3.10%2B-3776AB)](https://www.python.org/)

Vampyre turns source code, documents, structured project files, PDFs, images,
and media into a queryable knowledge graph for coding agents and humans. Code is
parsed locally. Semantic content is sent only to the explicitly selected LLM
backend after the credential-egress gate accepts it.

The distribution name is `graphifyy`; the installed command is `graphify`.
This is Vampyre's canonical standalone repository. Its source, documentation,
security policy, and releases are maintained here; no parent repository is an
update, support, or release channel. Project identity and branch policy are
recorded in [PROJECT.md](PROJECT.md).

## What It Produces

```text
graphify-out/
|-- graph.json             canonical NetworkX node-link graph
|-- graph.html             interactive browser view
|-- GRAPH_REPORT.md        hubs, communities, quality signals, and questions
|-- manifest.json          content hashes used for incremental detection
|-- .graphify_root         portable source-root marker
|-- .graphify_analysis.json
|-- .graphify_labels.json
`-- cache/                 AST and semantic caches
```

Vampyre supports simple graphs, directed graphs, and keyed directed
multigraphs. For production extraction, `--multigraph` preserves distinct
parallel relationships between the same endpoints. Existing graphs are sticky:
later extraction, update, watch, export, and MCP paths preserve the recorded
class and profile unless an explicit lossy class transition is requested.

## Requirements

- Python 3.10 or newer
- [uv](https://docs.astral.sh/uv/getting-started/installation/)

## Install

Install the exact 0.9.5 release:

```sh
uv tool install --force "graphifyy @ git+https://github.com/hypnwtykvmpr/vampyre.git@v0.9.5"
graphify --version
```

The [GitHub release](https://github.com/hypnwtykvmpr/vampyre/releases/tag/v0.9.5)
also provides Windows, macOS, and Linux bundles. Each bundle contains the same
universal wheel plus an OS-appropriate installer. That wheel is installed and
the real CLI is executed on the named OS before the release is published.

Development snapshots install from `main`:

```sh
uv tool install --force "graphifyy @ git+https://github.com/hypnwtykvmpr/vampyre.git@main"
```

Install an optional extra from the release tag by putting its name in brackets:

```sh
uv tool install --force "graphifyy[mcp] @ git+https://github.com/hypnwtykvmpr/vampyre.git@v0.9.5"
uv tool install --force "graphifyy[all] @ git+https://github.com/hypnwtykvmpr/vampyre.git@v0.9.5"
```

Available extras are `pdf`, `office`, `google`, `video`, `watch`, `mcp`,
`neo4j`, `svg`, `leiden`, `ollama`, `openai`, `gemini`, `kimi`, `anthropic`,
`bedrock`, `sql`, `postgres`, `dm`, `terraform`, `chinese`, and `all`. Azure
OpenAI uses the `openai` extra.

If the command is not on `PATH` after installation, run `uv tool update-shell`,
open a new terminal, and retry.

## Quick Start

Build a keyed directed multigraph without semantic LLM work for a code-only
project:

```sh
graphify extract . --multigraph
```

Build a mixed code/document graph with an explicit backend:

```sh
graphify extract . --multigraph --backend gemini
```

The backend needs its corresponding environment configuration. Examples include
`GEMINI_API_KEY`, `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`,
`AZURE_OPENAI_API_KEY` plus `AZURE_OPENAI_ENDPOINT`, `MOONSHOT_API_KEY`, AWS
credentials for Bedrock, or an explicit local `OLLAMA_BASE_URL`.

Query the result:

```sh
graphify query "what owns authentication?"
graphify path "RequestHandler" "DatabasePool"
graphify explain "RateLimiter"
graphify affected "UserModel" --depth 3
```

Export other views:

```sh
graphify export callflow-html
graphify export wiki --graph graphify-out/graph.json
graphify export obsidian --graph graphify-out/graph.json --dir ./vault
graphify export graphml --graph graphify-out/graph.json
```

`graphify --help` is the canonical runtime inventory. The maintained
[Command reference](docs/command-reference.md) groups every command by
workflow and records the safety-sensitive update flags. Format-specific export
examples are included there.

## Graph Classes And Updates

Fresh extraction defaults to a simple graph for compatibility. Select the class
explicitly when creating durable state:

```sh
graphify extract . --simple
graphify extract . --directed
graphify extract . --multigraph
```

`update` and `watch` require an existing graph with an explicit profile and a
valid scan-root marker. They refuse missing, corrupt, conflicting, or ambiguous
state before mutation; they do not initialize a rival graph or guess a class.

```sh
graphify update .
graphify update . --no-cluster
graphify watch .
```

Store graph state outside the scanned source tree with `--out`:

```sh
graphify extract ./Sources --out ./canonical --multigraph
graphify update ./Sources --out ./canonical
graphify update --out ./canonical
```

Use `--remap` when a full semantic re-extraction is required while preserving
the output root and graph profile. Use `--full` for a full rescan. `--force`
only accepts a verified non-empty shrink; it does not imply `--full`, bypass
caches, or allow an empty wipe.

```sh
graphify update ./Sources --out ./canonical --remap
graphify extract . --full --force
```

`update --repair-state` only restores a missing source-root marker after strict
graph and manifest validation. It does not infer graph class or repair corrupt
content.

## Agent Integrations

Register the bundled skill for the detected assistant:

```sh
graphify install
```

Choose a host explicitly, or install into the current project:

```sh
graphify install --platform codex
graphify install --platform agents --project
graphify claude install --project
graphify vscode install
```

Supported targets include Claude Code, Codex, Agent Skills, OpenCode, Kilo,
Cursor, Gemini CLI, GitHub Copilot CLI, VS Code Copilot Chat, Aider, Amp,
OpenClaw, Factory Droid, Trae, Hermes, Kimi Code, Kiro, Pi, Devin CLI, and
Google Antigravity. Run `graphify --help` for the exact install/uninstall
commands. In chat interfaces the invocation may be `/graphify`; Codex uses
`$graphify`; PowerShell users should invoke the CLI as `graphify` without a
leading slash.

The [Agent integration matrix](docs/agent-integrations.md) lists the supported
global and project-scoped installation command for every host.

Install repository hooks after building a graph:

```sh
graphify hook install
graphify hook status
```

The hooks update existing graph state after commits and checkouts. Reinstall
them after reinstalling or moving the uv-managed tool environment.

## Inputs And Egress

Tree-sitter and deterministic parsers handle supported source languages,
manifests, project files, SQL, and MCP configuration locally. Optional
converters handle Office files, Google Workspace pointers, PDFs, images, and
audio/video. The exact dependency set is declared in `pyproject.toml`.

Documents, papers, images, and transcripts can enter semantic extraction. Before
an outbound request, Vampyre:

1. contains every source path to the selected project root;
2. classifies credential-bearing filenames and content;
3. refuses credential-classified content without a bypass;
4. wraps accepted content as untrusted source data;
5. records content-free egress decisions for diagnostics.

Code files use local AST extraction in the normal pipeline and are not part of
the semantic document batch. A fully warm semantic cache can be reused without
provider credentials.

Use `.graphifyignore` for repository-specific exclusions. Matching follows Git
ignore semantics, including negation and anchored patterns. The output directory
is excluded from its own scan even when `GRAPHIFY_OUT` uses a custom name.

## MCP Server

Install the `mcp` extra, then start a local stdio server:

```sh
graphify serve graphify-out/graph.json
```

Loopback HTTP is available without authentication:

```sh
graphify serve graphify-out/graph.json --transport http --host 127.0.0.1 --port 8080
```

Non-loopback binds require all three security controls: an API key, a readable
TLS certificate/key pair, and valid Host-header configuration. Wildcard binds
also require one or more explicit `--allow-host` values.

```sh
graphify serve graphify-out/graph.json \
  --transport http \
  --host 0.0.0.0 \
  --port 8443 \
  --api-key "$GRAPHIFY_API_KEY" \
  --ssl-certfile ./cert.pem \
  --ssl-keyfile ./key.pem \
  --allow-host graph.example.test:8443
```

Multi-project access is bounded by repeatable `--allow-project-root` values.
GitHub-backed tools remain disabled unless `--enable-github-tools` and one fixed
`--github-repo OWNER/REPO` authority are supplied.

## Privacy

- No telemetry, analytics, or usage tracking is built in.
- Query logging is off by default. Set `GRAPHIFY_QUERY_LOG=1` for the default
  user-cache path or set it to an explicit file path.
- Full query responses are excluded from logs unless
  `GRAPHIFY_QUERY_LOG_RESPONSES=1` is explicitly set.
- `GRAPHIFY_QUERY_LOG_DISABLE=1` overrides every logging setting.
- Remote semantic backends receive accepted semantic source content. Select an
  explicit local backend when off-host egress is not acceptable.
- Local endpoints are still validated. A non-loopback Ollama URL emits a clear
  warning because it transmits the corpus off-host.

See [SECURITY.md](SECURITY.md) for the complete supported security model.

## Documentation Language

English documentation is authoritative for supported behavior, installation,
security, and release policy. Pre-standalone translations were retired because
they described older behavior and install sources. Their former paths therefore
do not represent current Vampyre guidance. Maintained translations may return
after review against a tagged release; until then, begin with this README and
the maintained references linked below.

## Development

Active development happens on `main`. `v9` is retained only as an exact
accepted-SHA compatibility mirror; it is not an install source or development
branch. `v8` is frozen historical state.

```sh
git clone https://github.com/hypnwtykvmpr/vampyre.git
cd vampyre
git checkout main
uv sync --all-extras --frozen
uv run --frozen graphify --version
```

Run the complete strict suite and local commit gates:

```sh
uv run --frozen pytest tests/ -n auto -q --tb=short
uv run --frozen pre-commit run --all-files
```

Hosted CI runs the full suite on Windows, macOS, and Linux with Python 3.10 and
3.13. Skips, deselections, xfail/xpass outcomes, warnings, lint findings, type
findings, and security findings fail the run.

Architecture, state, and extension guidance:

- [ARCHITECTURE.md](ARCHITECTURE.md)
- [How Vampyre works](docs/how-it-works.md)
- [Command reference](docs/command-reference.md)
- [Agent integration matrix](docs/agent-integrations.md)
- [Project identity and branch policy](PROJECT.md)
- [Security policy](SECURITY.md)
- [Release notes](docs/releases/0.9.5.md)

Vampyre retains the MIT license and original attribution required by its source
history. That history does not establish a current parent or support channel;
the canonical Vampyre project is this repository.
