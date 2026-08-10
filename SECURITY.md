# Security Policy

## Supported Version

| Version | Status |
| --- | --- |
| 0.9.5 | Supported |
| Earlier versions | Unsupported |

The canonical security policy and supported releases are maintained only at
https://github.com/hypnwtykvmpr/vampyre. Security fixes are delivered from this
repository and its `v0.9.5` release, not from a predecessor or parent project.

## Reporting A Vulnerability

Do not open a public issue containing vulnerability details, credentials, or a
working exploit. Use GitHub private vulnerability reporting when available, or
contact the repository owner privately. Include the affected version, impact,
reproduction steps, and the smallest safe proof needed to validate the report.

## Local And Remote Work

Vampyre performs code AST extraction, deterministic parsing, graph building,
clustering, analysis, querying, and local exports on the machine where it runs.
Network access is possible only through features that inherently require it:

- semantic extraction and community labeling through a selected LLM backend;
- explicit URL ingestion and video retrieval;
- optional Google Workspace conversion, PostgreSQL, Neo4j, or GitHub tools;
- optional MCP HTTP serving.

Documents, papers, images, and transcripts can be sent to the configured LLM
provider. Code files use local AST extraction in the normal pipeline. Before any
semantic payload leaves the machine, the egress gate blocks credential-bearing
paths and content, contains paths to the scan root, and records content-free
decisions. There is no waiver or per-run bypass for blocked credential content.

The selected backend determines data residency. Use an explicitly configured
local backend when source content must remain on-host. A non-loopback Ollama URL
is remote for this purpose and produces a warning.

## MCP Transport

The default MCP transport is local stdio. HTTP defaults to `127.0.0.1`.

Non-loopback HTTP binds fail closed unless all of the following are present:

1. an API key supplied by `--api-key` or `GRAPHIFY_API_KEY`;
2. a readable TLS certificate and key;
3. valid Host-header configuration, with explicit `--allow-host` entries for a
   wildcard bind.

HTTP uses constant-time credential comparison, DNS-rebinding protection, bounded
session/context caches, and configurable idle-session reaping. Multi-project
paths must remain under configured `--allow-project-root` values. GitHub tools
are disabled by default and, when enabled, require one fixed `OWNER/REPO` scope.

## Threat Controls

| Vector | Control |
| --- | --- |
| SSRF and redirects | URL validation allows HTTP(S), resolves destinations, blocks private/loopback/link-local/metadata targets, and revalidates redirects. |
| Oversized or hostile input | Download, graph, Office, archive, XML, and document paths enforce size/depth/member limits and safe parsers. |
| Path traversal and symlinks | Paths are resolved against explicit roots; out-of-root symlink targets and graph paths are refused. |
| Credential egress | Filename and content classification occurs before LLM dispatch; blocked files are not transmitted. |
| Prompt injection | Accepted source is neutralized, hash-delimited, and marked as untrusted data; the Claude CLI path is capability-confined. |
| HTML/text injection | Labels and user-controlled display values are bounded and escaped before HTML or MCP rendering. |
| Corrupt graph state | Strict decoding rejects conflicting class flags, profiles, endpoints, keys, provenance, and hyperedge membership before mutation. |
| Partial publication | Same-directory atomic writes, file locks, generation checks, and best-effort multi-file rollback protect canonical state. |
| Secret persistence | API keys are read from arguments/environment for the selected operation and are not intentionally written to graph state or query logs. |

## Query Logging

Query logging is disabled by default. It activates only when
`GRAPHIFY_QUERY_LOG` is set to `1`, `true`, `yes`, or an explicit file path.
Full graph responses require the additional explicit
`GRAPHIFY_QUERY_LOG_RESPONSES=1`. `GRAPHIFY_QUERY_LOG_DISABLE=1` always wins.

## Dependency And Release Security

The lock file is managed only with uv. CI blocks on dependency vulnerability
audit, static security analysis, lint, type checking, warnings, and the complete
cross-platform test suite. Release artifacts are built from a version-matching
tag, smoke-tested on Windows, macOS, and Linux, and published with SHA-256
checksums.
