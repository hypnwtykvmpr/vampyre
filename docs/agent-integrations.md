# Vampyre Agent Integration Matrix

Vampyre 0.9.5 ships assistant-specific skills and persistent project guidance.
The canonical source for these integrations is
<https://github.com/hypnwtykvmpr/vampyre>.

Install Vampyre itself before using this table:

The distribution name is `graphifyy`; the installed command is `graphify`.

```sh
uv tool install --force "graphifyy @ git+https://github.com/hypnwtykvmpr/vampyre.git@v0.9.5"
```

## Scope

- `graphify install --platform <name>` installs the selected host skill in its
  user scope.
- Add `--project` to install the selected integration in the current project
  when that host supports project-local skills or instructions.
- The direct project command, where listed, writes the host's persistent
  project instructions and supported hooks.
- `graphify uninstall` removes every detected integration. Add `--purge` only
  when `graphify-out/` should also be deleted.

## Platforms

| Host | User-scope skill | Project integration |
|---|---|---|
| Claude Code (macOS/Linux) | `graphify install --platform claude` | `graphify claude install --project` |
| Claude Code (Windows) | `graphify install --platform windows` | `graphify claude install --project` |
| CodeBuddy | `graphify install --platform codebuddy` | `graphify codebuddy install` |
| Codex | `graphify install --platform codex` | `graphify codex install` |
| OpenCode | `graphify install --platform opencode` | `graphify opencode install` |
| Kilo Code | `graphify install --platform kilo` | `graphify kilo install` |
| GitHub Copilot CLI | `graphify install --platform copilot` | `graphify copilot install` |
| VS Code Copilot Chat | `graphify vscode install` | `graphify vscode install` |
| Aider | `graphify install --platform aider` | `graphify aider install` |
| Amp | `graphify install --platform amp` | `graphify install --project --platform amp` |
| Agent Skills | `graphify install --platform agents` | `graphify install --project --platform agents` |
| OpenClaw | `graphify install --platform claw` | `graphify claw install` |
| Factory Droid | `graphify install --platform droid` | `graphify droid install` |
| Trae | `graphify install --platform trae` | `graphify trae install` |
| Trae CN | `graphify install --platform trae-cn` | `graphify trae-cn install` |
| Gemini CLI | `graphify install --platform gemini` | `graphify gemini install` |
| Cursor | `graphify install --platform cursor` | `graphify cursor install` |
| Google Antigravity | `graphify install --platform antigravity` | `graphify antigravity install` |
| Google Antigravity (explicit Windows target) | `graphify install --platform antigravity-windows` | `graphify antigravity install` |
| Hermes | `graphify install --platform hermes` | `graphify hermes install` |
| Kimi Code | `graphify install --platform kimi` | `graphify install --project --platform kimi` |
| Kiro IDE/CLI | `graphify install --platform kiro` | `graphify kiro install` |
| Pi coding agent | `graphify install --platform pi` | `graphify pi install` |
| Devin CLI | `graphify install --platform devin` | `graphify devin install` |

Run `graphify install --help` for the live platform identifier list. Run
`graphify --help` for host-specific uninstall commands.

## Invocation

- Chat-oriented hosts normally invoke the skill as `/graphify`.
- Codex invokes it as `$graphify`.
- PowerShell invokes the installed command as `graphify` without a leading
  slash.

The installed guidance directs the assistant to query existing
`graphify-out/graph.json` state before broad source searches. Hosts with hook
support also receive the corresponding query-first hook. Re-run the integration
command after moving or reinstalling the uv-managed Vampyre tool environment.

## Repository Hooks

Repository hooks are independent of assistant integration:

```sh
graphify hook install
graphify hook status
graphify hook uninstall
```

They update an existing graph after commits and checkouts. They do not create
initial graph state; run `graphify extract` first.
