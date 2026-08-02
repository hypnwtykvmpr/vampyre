# Vampyre Fork Notes

Vampyre is an independent fork of upstream Graphify. It preserves upstream's
useful foundation while carrying a production MultiDiGraph implementation and
fork-specific hardening. Upstream is credited as the source project; it is a
read-only reference, not this fork's release channel.

## Installing This Fork

The PyPI package `graphifyy` installs upstream Graphify's published package, not
this fork. To install this fork, use the GitHub branch or a local checkout:

```bash
# From GitHub:
uv tool install --force "graphifyy @ git+https://github.com/hypnwtykvmpr/vampyre.git@v9"

# Or from a clone of this repository:
uv tool install --force .
```

Then install the assistant skill as usual:

```bash
graphify install
```

Optional extras can be installed from a local checkout with commands such as
`uv tool install --force ".[office]"`. From GitHub, use a direct reference such
as:

```bash
uv tool install --force "graphifyy[office] @ git+https://github.com/hypnwtykvmpr/vampyre.git@v9"
```

The `graphifyy` registry package is upstream Graphify, not Vampyre. Vampyre is
installed from this repository's `v9` branch or from a local checkout.

## Branches

- `main` is the sole integration branch.
- `v9` mirrors `main` as the active version line.
- `v8` is frozen at the last pre-v9 fork state and is never advanced by current
  work.
- `upstream-v8-base` is a clean fork-side reference to upstream `v8` for future
  comparison.
- `upstream` remains read-only. Importing future upstream work is a deliberate
  human decision, not an automatic tracking policy.

## MultiDiGraph Scope

The fork adds opt-in `--multigraph` support for preserving parallel relationships
between the same node pair as keyed edges. The default simple-graph path remains
available for compatibility, and `--simple` is the explicit lossy downgrade from
an existing multigraph profile.

The implemented multigraph path covers build/load, dedup/remap, query/path
display, MCP surfaces, exports, visualizations, watch/update, cache reuse, and
global graph merge/recovery. Algorithmic consumers that need simple graphs use
explicit projections so parallel edges do not inflate metrics by accident.

Producer widening is intentionally conservative and evidence-driven. This fork
currently preserves additional parallel detail for distinct call sites, cross-file
calls that coexist with other relations, and JS/TS dynamic import sites. Remaining
candidate producers should be widened only when diagnostics and focused
regression tests show useful signal rather than extraction noise.

## Upstream Posture

Upstream Graphify is the source project. Local fork changes should continue to be
small, documented, and easy to rebase. When upstream changes touch the same files,
compare both sides directly, preserve useful upstream behavior, and document any
intentional divergence.

## Integrating Upstream Work

Upstream changes are evaluated for objective value and compatibility. Nothing is
imported merely to keep branch counters even, and useful work is not discarded
merely because it originated upstream. Every conflict is compared directly;
Vampyre's capabilities and acceptance contract must survive the integration.

1. Fetch and record the exact upstream freeze SHA being evaluated.
2. Compare every upstream commit and every conflict against the fork contract.
3. Replay only approved behavior onto `main`; do not advance frozen `v8`.
4. Run the complete zero-waiver verification battery.
5. Publish one reviewed commit to `main` and align `v9` to it.
