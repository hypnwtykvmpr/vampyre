# Vampyre Command Reference

This is the maintained command map for Vampyre 0.9.5. The canonical source is
the CLI shipped by this repository:

```sh
graphify --help
```

Use `graphify <command> --help` when that command exposes detailed parser help.
The canonical source and release channel is
<https://github.com/hypnwtykvmpr/vampyre>.

## Installation And Version

The distribution name is `graphifyy`; the installed command is `graphify`.

```sh
uv tool install --force "graphifyy @ git+https://github.com/hypnwtykvmpr/vampyre.git@v0.9.5"
graphify --version
graphify --help
```

Optional capabilities use the same immutable source with an extra name:

```sh
uv tool install --force "graphifyy[pdf] @ git+https://github.com/hypnwtykvmpr/vampyre.git@v0.9.5"
uv tool install --force "graphifyy[mcp] @ git+https://github.com/hypnwtykvmpr/vampyre.git@v0.9.5"
uv tool install --force "graphifyy[all] @ git+https://github.com/hypnwtykvmpr/vampyre.git@v0.9.5"
```

Available extras are `pdf`, `office`, `google`, `video`, `watch`, `mcp`,
`neo4j`, `svg`, `leiden`, `ollama`, `openai`, `gemini`, `kimi`, `anthropic`,
`bedrock`, `sql`, `postgres`, `dm`, `terraform`, `chinese`, and `all`. Azure
OpenAI uses the `openai` extra.

## Build And Maintain Graph State

| Command | Purpose |
|---|---|
| `graphify extract <path>` | Build graph state from code and optional semantic inputs. |
| `graphify update [<path>]` | Refresh an existing graph while preserving its recorded class and profile. |
| `graphify watch [<path>]` | Watch the saved source tree and update existing graph state. |
| `graphify cluster-only <path>` | Recluster an existing graph and regenerate its report. |
| `graphify label <path>` | Name or rename graph communities with the selected backend. |
| `graphify check-update [<path>]` | Check the pending-update signal without mutating graph state. |
| `graphify add <url>` | Fetch content into `./raw` and update the graph. |

Select the initial graph class explicitly for durable state:

```sh
graphify extract . --simple
graphify extract . --directed
graphify extract . --multigraph
```

Important update controls:

| Option | Contract |
|---|---|
| `--out DIR` | Read and write graph state below the selected output root. |
| `--no-cluster` | Write extracted graph content without community detection. |
| `--no-viz` | Skip browser visualization generation. |
| `--remap` | Re-extract fully while preserving the output root and graph profile. |
| `--repair-state` | Restore only a missing scan-root marker after strict state validation. |
| `--full` | Perform a full rescan. |
| `--force` | Accept only a verified non-empty shrink; never clear caches, imply `--full`, or permit an empty wipe. |

Use `graphify extract . --full --force` for a full rebuild that may legitimately
shrink.

## Query And Analysis

| Command | Purpose |
|---|---|
| `graphify query "<question>"` | Traverse graph context using breadth-first search; add `--dfs` for depth-first traversal. |
| `graphify path "A" "B"` | Find the shortest path between two nodes. |
| `graphify explain "X"` | Explain one node and its neighbors. |
| `graphify affected "X"` | Traverse reverse relationships to identify impacted nodes. |
| `graphify benchmark [graph.json]` | Measure graph-context token reduction against a naive corpus load. |
| `graphify diagnose multigraph` | Report same-endpoint relationship collapse risk. |
| `graphify tree` | Generate an interactive file/symbol hierarchy. |

### Query Budgets

`graphify query --budget N` bounds the response. The queried symbol is rendered
first, so it can never be crowded out of its own answer by higher-degree
neighbours, and the budget admits **complete evidence records only** — a `NODE`
or `EDGE` record is never split, and a relationship is never emitted without
both of its endpoint nodes. When records are dropped, the truncation line
reports how many nodes and keyed relationships were omitted.

The primary record is the queried symbol together with its first canonical
relationship and that relationship's counterpart node. When the queried symbol
has no relationships in the traversed subgraph, the primary record is the
symbol's node alone.

A budget too small to hold that primary record is refused rather than answered
partially. The reported minimum is computed from the actual graph and question,
so it varies by query — use the value the command prints, not a fixed number:

```console
$ graphify query "extract" --budget 1
error: --budget 1 is too small for one complete evidence record; retry with --budget N or higher
$ echo $?
2
```

Earlier releases returned a truncated apparent success with exit `0` for the
same request. Scripts that check the exit status will now see `2` and can retry
at the reported minimum. Over MCP the same condition returns
`isError: true` with structured content
`{"code": "insufficient_budget", "required_minimum": <int>}`.

Query results can feed deterministic work memory:

```sh
graphify save-result --question "..." --answer "..." --outcome useful
graphify reflect
```

## Import, Merge, And Global State

| Command | Purpose |
|---|---|
| `graphify clone <github-url>` | Clone a repository into Vampyre's local repository cache. |
| `graphify merge-graphs <g1> <g2>` | Merge graph files into one cross-repository graph. |
| `graphify merge-driver <base> <current> <other>` | Run the graph-aware Git merge driver. |
| `graphify global add <graph.json>` | Add or update a tagged project in the user-global graph. |
| `graphify global remove <tag>` | Remove one tagged project from the user-global graph. |
| `graphify global list` | List repositories represented in the user-global graph. |
| `graphify global path` | Print the global graph path. |

When two inputs reuse the same default repository tag, supply an explicit
`--as <tag>` rather than allowing ambiguous ownership.

## Export

Use `graphify export <format>` for HTML, wiki, Obsidian, GraphML, SVG, Neo4j,
and call-flow output. Common forms include:

```sh
graphify export callflow-html
graphify export wiki --graph graphify-out/graph.json
graphify export obsidian --graph graphify-out/graph.json --dir ./vault
graphify export graphml --graph graphify-out/graph.json
```

Exports do not redefine the canonical graph class. A lossy representation must
be explicitly selected where one is supported.

## Providers, Pull Requests, And MCP

| Command | Purpose |
|---|---|
| `graphify provider add` | Register a named custom model provider. |
| `graphify provider list` | List custom providers. |
| `graphify provider show` | Show one provider without exposing secrets. |
| `graphify provider remove` | Remove one provider. |
| `graphify prs` | Show open pull requests with CI, review, worktree, and optional graph-impact context. |
| `graphify serve [graph.json]` | Serve graph tools over MCP stdio or secured HTTP. |

`graphify serve` defaults to stdio. Loopback HTTP can run without
authentication. Non-loopback HTTP requires an API key, TLS certificate/key,
and valid Host-header configuration; wildcard binds also require explicit
`--allow-host` values. See [the security policy](../SECURITY.md).

## Installation Integrations And Hooks

| Command | Purpose |
|---|---|
| `graphify install [--platform P]` | Install the bundled skill for one supported assistant. |
| `graphify uninstall` | Remove detected Vampyre integrations; `--purge` also removes `graphify-out/`. |
| `graphify hook install` | Install graph update hooks for commits and checkouts. |
| `graphify hook status` | Show hook installation state. |
| `graphify hook uninstall` | Remove installed graph hooks. |

Host-specific commands and project-scope behavior are listed in the
[agent integration matrix](agent-integrations.md).
