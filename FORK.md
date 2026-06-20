# Vampyre Fork Notes

This fork tracks upstream Graphify closely while carrying local changes for
MultiDiGraph evaluation and development.

## Installing This Fork

The PyPI package `graphifyy` installs upstream Graphify's published package, not
this fork. To install this fork, use the GitHub branch or a local checkout:

```bash
# From GitHub:
uv tool install --force git+https://github.com/hypnwtykvmpr/vampyre.git@v8

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
uv tool install --force "graphifyy[office] @ git+https://github.com/hypnwtykvmpr/vampyre.git@v8"
```

Use `uv tool install graphifyy` only when you intentionally want upstream's PyPI
release.

## Branches

- `v8` is the active fork branch and is rebased on upstream `v8` when useful
  upstream changes land.
- `main` mirrors the active fork line for GitHub visitors who expect a `main`
  branch.
- `upstream-v8-base` is a clean fork-side reference to upstream `v8` for future
  rebases and comparison.
- `upstream` remains read-only unless contribution work is explicitly reopened.

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

## Upstream Sync Workflow (standing policy)

Keep a **clean linear history**: rebase the fork stack onto the newer upstream and
**force-push** the result. Origin does not block force-pushes (no branch
protection; the only ruleset is Copilot review), so there is no need for the old
`-s ours` reconcile-merge workaround — that only accumulated duplicate commits in
ancestry. Flatten every sync instead.

1. `git fetch upstream` — if `upstream/v8` has not advanced, just fast-forward-push
   any pending fork commits and stop.
2. Rebase the fork's clean commit stack onto `upstream/v8` (replay only the fork
   delta — rebase the pre-merge tip, not a branch carrying old reconcile merges).
   Hand-weave conflicts: keep upstream's improvements AND the fork's
   multigraph/security delta. A commit that upstream has since merged (e.g. our
   dependabot PR) will drop as empty — that is expected.
3. Run the full suite env-stripped — it must be **0 failures, 0 skips** (the
   fork's `conftest.py` treats skips as failures). Confirm `git diff` against the
   prior deployed tree is empty unless an intentional change was made.
4. Force-push the flattened linear history:
   `git push --force-with-lease origin v8:v8 main:main`.
   The agent's command guard blocks force-push, so a human runs this step (from a
   normal terminal). Verify `origin == local` afterward.
5. Trim the temporary work branch.
