# Architecture

Vampyre is a Python package, command-line application, agent-skill bundle, and
optional MCP server. The `graphify` CLI is the shared entry point; agent
integrations invoke the same implementation rather than maintaining a separate
graph engine.

## Pipeline

```text
detect -> local extraction/conversion -> semantic egress -> build
       -> cluster -> analyze -> transactional publish -> query/export/serve
```

Incremental paths reuse the same state contracts:

```text
manifest + source marker + graph profile
       -> detect changes -> stable extraction -> merge/prune
       -> validate -> transactional publish
```

## Major Modules

| Module | Responsibility |
| --- | --- |
| `__main__.py` | CLI dispatch and full extraction orchestration |
| `detect.py` | file classification, ignore rules, manifests, content snapshots |
| `extract.py`, `extractors/` | deterministic language and format extraction |
| `llm.py`, `egress.py` | semantic backend selection, credential egress gate, untrusted-source payloads |
| `build.py`, `edge_identity.py` | node reconciliation, keyed edge identity, graph construction |
| `graph_state.py`, `graph_loader.py` | canonical serialized-state codec and read modes |
| `cluster.py`, `projections.py` | deterministic community detection and explicit simple projections |
| `analyze.py`, `report.py` | graph metrics, quality analysis, Markdown report |
| `export.py`, `wiki.py`, `callflow_html.py` | JSON, HTML, wiki, Obsidian, GraphML, SVG, Neo4j, and call-flow outputs |
| `watch.py`, `update_state.py`, `hooks.py` | strict incremental updates and Git lifecycle integration |
| `persistence.py` | locks, atomic writes, leases, rollback, and pending generations |
| `serve.py`, `security.py` | CLI/MCP queries, HTTP transport, path and network boundaries |
| `cache.py`, `provenance.py`, `semantic_schema.py` | identity-aware caches and producer provenance |

## Canonical Graph State

`graph.json` uses NetworkX node-link data with one explicit profile:

- `simple`: undirected `Graph`
- `digraph`: directed `DiGraph`
- `multidigraph`: keyed directed `MultiDiGraph`

Undirected multigraphs are intentionally not representable. A serialized state
must have consistent boolean class flags, profile, node IDs, edge endpoints,
keyed edge identities, provenance, and hyperedge members. Current mutation paths
use strict decoding and refuse malformed state. Read-only compatibility paths
may normalize legacy shapes only when the interpretation is lossless.

The canonical profile, metadata, custom profile fields, hyperedges, edge keys,
parallel relationships, and producer provenance must survive every stateful
path. Algorithms that need simple topology use explicit projections rather than
silently changing the stored graph class.

## Identity And Determinism

Node IDs are path-qualified and generated from canonical POSIX-style source
identities. AST and semantic producers carry independent schema versions, and
their caches are namespaced by the inputs that affect identity.

Ordering is canonicalized before serialization and before order-sensitive graph
algorithms. Ambiguous aliases are skipped with diagnostics rather than bound to
an arbitrary first writer. LLM-authored content is the explicit nondeterministic
boundary; structural extraction, merge, clustering inputs, state serialization,
and repeated no-change updates are expected to be deterministic.

## Persistence

The selected output root owns all graph state, locks, queues, caches, reports,
markers, and temporary files. Atomic replacement occurs in the destination
directory. Multi-file publication captures prior state and performs best-effort
rollback across every captured file before reporting rollback failures.

Source bytes are hashed before extraction and checked again before manifest
advancement. A source change during extraction refuses publication instead of
marking mixed content current. Lease and generation checks prevent an older
publisher from acknowledging or overwriting newer work.

## Security Boundaries

- URL fetches validate scheme, destination, redirects, and response size.
- Source and graph paths are canonicalized and contained to configured roots.
- Credential-classified semantic content is blocked before provider dispatch.
- The confined Claude CLI backend runs without project tools or inherited
  repository authority.
- HTTP MCP is loopback-only without authentication; non-loopback binds require
  API-key auth and TLS, with DNS-rebinding and Host-header controls.
- Multi-project and GitHub MCP authority are explicit allowlists.

See [SECURITY.md](SECURITY.md) for operational details.

## Extending Extractors

Language extractors return `nodes`, `edges`, and optional `hyperedges` using the
shared schema. New or moved extractors must preserve facade imports, registry
identity, source-root handling, provenance, keyed relationship identity, and
deterministic output. Follow [the extractor migration guide](graphify/extractors/MIGRATION.md)
and add non-vacuous fixtures for both expected output and edge cases.

## Verification

```sh
uv run --frozen pytest tests/ -n auto -q --tb=short
uv run --frozen pre-commit run --all-files
```

CI runs the complete strict suite on Windows, macOS, and Linux. The release
workflow separately builds the wheel and installs it in isolated uv tool roots
on all three operating systems before publishing artifacts.
