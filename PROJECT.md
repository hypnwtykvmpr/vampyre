# Project Identity And Branches

Vampyre is an independent standalone project. Its canonical source,
documentation, security policy, and releases are maintained at
https://github.com/hypnwtykvmpr/vampyre. It has no parent repository and does
not inherit updates, support, release authority, or maintenance decisions from
another project.

Vampyre originated from the MIT-licensed Graphify codebase and retains its
license and attribution. That history is provenance, not a current maintenance
relationship.

External changes may still be evaluated as ordinary third-party proposals. They
are accepted only when evidence shows that they improve Vampyre without
regressing its behavior. Branch counters and source similarity are not release
criteria.

## Branches

- `main` is the sole working, integration, default, and publication branch.
- `v9` is an exact accepted-SHA compatibility mirror. It is not a development
  branch or installation source.
- `v8` is frozen at `21a9f1f` as historical state and is never advanced.

Stable installations use an immutable version tag such as `v0.9.5`.
Development snapshots use `main`.

## Documentation

English documentation is authoritative for supported behavior, installation,
security, and release policy. Pre-standalone translations were retired for
0.9.5 because they still described the former project identity, support links,
feature set, and `v9` development installs. They must not be used as current
Vampyre guidance. Maintained translations may return after they are reviewed
against the released English documentation and its immutable version tag.

## Releases

A release is built only from a tag that exactly matches the version in
`pyproject.toml`. The universal wheel is installed and exercised on Windows,
macOS, and Linux before GitHub publishes the release. OS-labeled bundles contain
that tested wheel and an installer; they are not separate or divergent builds.
Every asset is covered by `SHA256SUMS`.

## Graph Guarantees

Vampyre supports explicit simple, directed, and keyed directed multigraph
profiles. Class, profile, metadata, hyperedges, edge keys, provenance, valid
parallel relationships, and producer identity survive every stateful and
serialized path. Lossy transitions are explicit, warned, and never inferred by
an update.

Algorithmic consumers that need simple topology use explicit projections. This
keeps metrics meaningful without mutating or collapsing the canonical graph.
