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

Origin enforces branch protection: **force-pushes are rejected**. Because rebasing
the fork stack onto a newer upstream rewrites commit SHAs, the rebased branch
diverges from `origin` and a plain push is a non-fast-forward. Do **not** try to
force it — reconcile so the push stays a fast-forward:

1. `git fetch upstream` — if `upstream/v8` has not advanced, there is nothing to
   rebase (only push any pending fork commits as a normal fast-forward).
2. Rebase the fork's clean commit stack onto `upstream/v8` (replay only the fork
   delta; drop any prior reconcile-merge commits). Hand-weave conflicts: keep
   upstream's improvements AND the fork's multigraph/security delta.
3. Re-apply post-stack fork commits (e.g. dependency security pins) and run the
   full suite env-stripped — it must be **0 failures, 0 skips** (see `conftest.py`).
4. Reconcile with origin so the push fast-forwards, never force:
   `git merge -s ours origin/v8` records origin's tip as a parent while keeping the
   rebased tree verbatim (`git diff <rebased-tip> HEAD` must be empty).
5. `git push origin v8:v8 main:main` (fast-forward). Verify `origin == local`.
6. Trim the temporary work branch.
