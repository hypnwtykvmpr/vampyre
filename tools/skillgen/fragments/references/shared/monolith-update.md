## For --update (incremental re-extraction)

Do not run a hand-written NetworkX merge. Validate the existing graph state and classify the changed files first:

```bash
"$(cat graphify-out/.graphify_python)" -c "
import json
from pathlib import Path
from graphify.detect import detect_incremental
from graphify.update_state import resolve_update_context

context = resolve_update_context(Path('INPUT_PATH'))
incremental = detect_incremental(
    context.scan_root,
    manifest_path=str(context.manifest_path),
)
changed = [path for paths in incremental.get('new_files', {}).values() for path in paths]
code_exts = {'.py','.ts','.js','.go','.rs','.java','.cpp','.c','.rb','.swift','.kt','.cs','.scala','.php','.cc','.cxx','.hpp','.h','.kts','.lua','.toc','.f','.f90','.f95','.f03','.f08'}
print(json.dumps({
    'scan_root': str(context.scan_root),
    'output_root': str(context.output_root),
    'graph_type': context.graph_type,
    'changed': changed,
    'deleted': incremental.get('deleted_files', []),
    'code_only': all(Path(path).suffix.lower() in code_exts for path in changed),
}, indent=2))
"
```

- If there are no changed or deleted files, stop. Do not rewrite graph state.
- If every changed file is code (or the update contains only deletions), run `graphify update INPUT_PATH` and stop. The package command owns locking, keyed-edge preservation, source-generation checks, backups, and atomic graph/manifest publication.
- If a document, paper, image, or video changed, do a full Steps 2-9 rebuild of the resolved `scan_root` instead of an ad-hoc incremental merge. Inherit `IS_DIRECTED` and `IS_MULTIGRAPH` from the validated `graph_type`; a missing or conflicting profile is an error, never a reason to choose simple mode.
- If the validated output is not the default local `graphify-out`, use `graphify update INPUT_PATH --out OUTPUT_ROOT` for code-only work. For semantic work, use `graphify update INPUT_PATH --out OUTPUT_ROOT --remap` with a configured backend; this monolithic workflow must not redirect its hard-coded temporary files into a different canonical output.

This deliberately trades semantic incremental efficiency for state safety on monolithic hosts. It preserves semantic capability through a full rebuild and never publishes a partial changed-file graph.
