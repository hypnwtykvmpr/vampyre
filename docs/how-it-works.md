# How Vampyre Works

## 1. Detect And Classify

Vampyre walks the requested source root without following external symlink
targets. It combines Git-compatible `.gitignore` and `.graphifyignore` rules,
classifies supported inputs, and records content hashes in `manifest.json`.
Custom output directories are excluded from their own scans.

Incremental detection compares content, not only timestamps. Source bytes are
snapshotted again before publication, so files changed during extraction cannot
be committed as a mixed-generation graph.

## 2. Extract Local Structure

Code and structured project files use tree-sitter or deterministic parsers.
These paths emit symbols, imports, calls, type references, manifests, and other
explicit relationships without an LLM. CPU-bound AST work can use multiple
processes while preserving stable output ordering.

Optional local converters handle Office files, PDFs, Google Workspace pointer
files, images, and media. Audio/video transcription uses the optional video
dependencies and caches completed work by content identity.

## 3. Gate Semantic Egress

Code files are not sent to the LLM semantic extractor. If a corpus contains
only code files, Pass 3 is skipped entirely; semantic extraction is reserved
for docs, papers, images, and transcripts.

Documents, papers, images, and transcripts can enter semantic extraction. For
every candidate, Vampyre validates root containment and applies credential path
and content policy before preparing an outbound payload. Credential-classified
content is reported and omitted; it cannot be overridden per run.

Accepted text is neutralized and wrapped in hash-stamped
`<untrusted_source>` blocks. Backends include Gemini, Kimi, Anthropic, OpenAI and
compatible endpoints, DeepSeek, Azure OpenAI, Bedrock, Ollama, and a confined
Claude CLI path. Backend choice is explicit or detected from configured
credentials. A warm cache avoids another request when the semantic identity,
backend, model, schema, and content all match.

## 4. Reconcile And Build

Extractor results are validated before graph construction. Nodes are reconciled
by canonical identity. Ambiguous aliases are skipped instead of being attached
to an arbitrary candidate. Edge identity includes the fields required to retain
valid parallel relationships and distinct call sites.

The graph profile is one of:

- `simple`: undirected, one edge per endpoint pair;
- `digraph`: directed, one edge per ordered endpoint pair;
- `multidigraph`: directed, keyed parallel edges.

Hyperedges remain graph-level records with two or more live members. Profile,
custom metadata, provenance, keys, and hyperedges are part of the serialized
state contract, not optional decorations.

## 5. Cluster And Analyze

Community detection receives a canonically ordered graph projection. Native
Leiden is used when its optional dependency and Python version are supported;
the deterministic NetworkX fallback is used otherwise. Directed and parallel
state is preserved in storage while algorithms that require simple topology use
explicit projections.

Analysis computes community membership, cohesion, distinct-neighbor hubs,
surprising cross-community relationships, and suggested graph questions.
Community labels can be deterministic placeholders or generated through the
selected LLM backend.

## 6. Publish Transactionally

The graph, manifest, scan-root marker, analysis, labels, report, and related
state are written under one selected output root. Same-directory temporary
files, atomic replacement, output locks, generation checks, and multi-file
rollback prevent partial or stale publishers from silently becoming canonical.

No-change updates are byte-stable after settlement. `update` and `watch` refuse
unprofiled or ambiguous existing state before mutation. `--out` separates the
source root from the storage root without changing node IDs or cache identity.

## 7. Query, Export, And Serve

The graph can be queried through `query`, `path`, `explain`, and `affected`, or
served over MCP stdio/HTTP. Parallel relationships are reported as bounded
relationship envelopes rather than silently selecting one edge.

Exports include HTML, call-flow HTML, Markdown wiki, Obsidian, GraphML, SVG, and
Neo4j forms. Exporters track their own generated files and do not delete
unowned user content.

## Serialized Shape

`graph.json` is canonical NetworkX node-link JSON. At minimum, nodes carry an
`id`, `label`, source identity, and file type. Edges carry `source`, `target`,
`relation`, confidence, provenance, and a key when the profile is a
`multidigraph`. Graph metadata carries the explicit profile and identity schema.
Hyperedges live in graph metadata with stable IDs and live member IDs.

The strict codec rejects malformed current state. A separate read-only legacy
mode can normalize only shapes that have one lossless interpretation.

See [ARCHITECTURE.md](../ARCHITECTURE.md) for module ownership and
[SECURITY.md](../SECURITY.md) for the trust boundaries.
