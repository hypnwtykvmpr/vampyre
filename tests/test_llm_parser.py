"""Tests for `_parse_llm_json` robustness and the `_call_claude_cli`
subprocess argv shape introduced in the hollow-response fix.

These tests cover:
- The four parser failure modes described in PR #1062
- Extraction instructions delivered in the user turn (Claude Code >= 2.1)
- The GRAPHIFY_CLAUDE_CLI_MODEL env-var passthrough
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from graphify import llm
from graphify.claude_cli import _REQUIRED_HELP_FLAGS, _validated_executable
from graphify.ids import make_id
from graphify.semantic_schema import (
    EDGE_FIELDS,
    EDGE_RELATIONS,
    HYPEREDGE_FIELDS,
    NODE_FIELDS,
    PROMPT_SCHEMA_VERSION,
    SemanticSchemaError,
    render_semantic_schema,
    validate_semantic_fragment,
    validate_semantic_source_paths,
)


def test_rendered_semantic_schema_is_nonempty_versioned_and_field_complete():
    rendered = render_semantic_schema()
    example = json.loads(rendered.rsplit("\n", 1)[-1])

    assert PROMPT_SCHEMA_VERSION in rendered
    assert set(example["nodes"][0]) == set(NODE_FIELDS)
    assert set(example["edges"][0]) == set(EDGE_FIELDS)
    assert set(example["hyperedges"][0]) == set(HYPEREDGE_FIELDS)


def test_semantic_schema_accepts_records_without_optional_rationale_or_context():
    fragment = {
        "nodes": [
            {
                "id": node_id,
                "label": node_id.upper(),
                "file_type": "concept",
                "source_file": "docs/design.md",
            }
            for node_id in ("alpha", "beta", "gamma")
        ],
        "edges": [
            {
                "source": "alpha",
                "target": "beta",
                "relation": "references",
                "confidence": "EXTRACTED",
                "confidence_score": 1.0,
                "source_file": "docs/design.md",
            }
        ],
        "hyperedges": [
            {
                "id": "design_flow",
                "label": "Design Flow",
                "nodes": ["alpha", "beta", "gamma"],
                "relation": "participate_in",
                "confidence": "EXTRACTED",
                "confidence_score": 1.0,
                "source_file": "docs/design.md",
            }
        ],
    }

    assert validate_semantic_fragment(fragment) == fragment


def test_semantic_schema_accepts_digest_bearing_canonical_ids():
    """Multipart producer IDs remain valid lowercase semantic identifiers."""
    node_id = make_id("pkg", "python", "demo")
    fragment = {
        "nodes": [
            {
                "id": node_id,
                "label": "Demo",
                "file_type": "code",
                "source_file": "demo.py",
            }
        ],
        "edges": [],
        "hyperedges": [],
    }

    assert validate_semantic_fragment(fragment)["nodes"][0]["id"] == node_id


# ---------- _parse_llm_json: the four canonical failure modes ----------


def test_preamble_then_fence_is_parsed():
    """Claude often prefixes the JSON with a short preamble before the
    ```json fence. The original parser only stripped fences at offset 0,
    so any preamble caused json.loads to fail and the chunk to be
    dropped as a hollow response. The robust parser handles fences
    anywhere in the text."""
    raw = 'Here are the extracted entities:\n\n```json\n{"nodes": [{"id": "a"}], "edges": []}\n```'
    result = llm._parse_llm_json(raw)
    assert result["nodes"] == [{"id": "a"}]
    assert result["edges"] == []


def test_prose_wrapped_json_without_fence_is_parsed():
    """Some models return prose around bare JSON with no markdown fence.
    The balanced-brace fallback extracts the first complete object."""
    raw = 'The extracted graph is {"nodes": [{"id": "b"}], "edges": []}. Hope this helps!'
    result = llm._parse_llm_json(raw)
    assert result["nodes"] == [{"id": "b"}]


def test_raw_json_still_works():
    """Regression: clean JSON input (the original happy path) must keep
    parsing exactly as before."""
    raw = '{"nodes": [], "edges": [], "hyperedges": []}'
    result = llm._parse_llm_json(raw)
    assert result == {"nodes": [], "edges": [], "hyperedges": []}


def test_total_refusal_returns_empty_fragment():
    """When the model refuses or returns unrelated prose, the parser
    must degrade gracefully — return the empty fragment so the hollow
    detector takes over, never raise."""
    raw = "I cannot extract structured data from this content."
    result = llm._parse_llm_json(raw)
    assert result == {"nodes": [], "edges": [], "hyperedges": []}


# ---------- _parse_llm_json: secondary cases worth pinning ----------


def test_fence_with_uppercase_language_tag():
    raw = '```JSON\n{"nodes": [{"id": "x"}], "edges": []}\n```'
    result = llm._parse_llm_json(raw)
    assert result["nodes"] == [{"id": "x"}]


def test_fence_without_closing_backticks():
    """Truncated response: the model started the fence but ran out of
    tokens before closing it. We should still recover the JSON body."""
    raw = '```json\n{"nodes": [{"id": "y"}], "edges": []}'
    result = llm._parse_llm_json(raw)
    assert result["nodes"] == [{"id": "y"}]


def test_empty_response_returns_empty_fragment():
    assert llm._parse_llm_json("") == {"nodes": [], "edges": [], "hyperedges": []}


def _valid_semantic_fragment() -> dict:
    return {
        "nodes": [
            {
                "id": "docs_design_runner",
                "label": "Runner",
                "file_type": "concept",
                "source_file": "docs/design.md",
                "source_location": None,
                "source_url": None,
                "captured_at": None,
                "author": None,
                "contributor": None,
                "rationale": None,
            }
        ],
        "edges": [],
        "hyperedges": [],
        "input_tokens": 1,
        "output_tokens": 2,
    }


def test_semantic_schema_rejects_unknown_relation_with_stable_path():
    fragment = _valid_semantic_fragment()
    fragment["edges"] = [
        {
            "source": "docs_design_runner",
            "target": "docs_design_worker",
            "relation": "hallucinated_relation",
            "confidence": "INFERRED",
            "confidence_score": 0.85,
            "source_file": "docs/design.md",
            "source_location": None,
            "weight": 1.0,
            "context": None,
        }
    ]

    with pytest.raises(
        SemanticSchemaError,
        match=r"^semantic schema violation at edges\[0\]\.relation: unknown relation$",
    ):
        validate_semantic_fragment(fragment)


def test_semantic_schema_rejects_unknown_fields_and_noncanonical_scores():
    fragment = _valid_semantic_fragment()
    fragment["nodes"][0]["surprise"] = "uncontracted"
    with pytest.raises(SemanticSchemaError, match="unknown field"):
        validate_semantic_fragment(fragment)

    fragment = _valid_semantic_fragment()
    fragment["edges"] = [
        {
            "source": "docs_design_runner",
            "target": "docs_design_worker",
            "relation": "references",
            "confidence": "INFERRED",
            "confidence_score": 0.8,
            "source_file": "docs/design.md",
            "source_location": None,
            "weight": 1.0,
            "context": None,
        }
    ]
    with pytest.raises(SemanticSchemaError, match="confidence_score"):
        validate_semantic_fragment(fragment)


def test_provider_response_validation_uses_the_canonical_schema():
    raw = json.dumps(_valid_semantic_fragment())
    assert llm._parse_semantic_response(raw) == _valid_semantic_fragment()

    invalid = _valid_semantic_fragment()
    invalid["nodes"][0]["file_type"] = "unknown"
    with pytest.raises(SemanticSchemaError, match=r"nodes\[0\]\.file_type"):
        llm._parse_semantic_response(json.dumps(invalid))


def test_semantic_schema_rejects_source_paths_not_sent_to_model():
    fragment = _valid_semantic_fragment()
    validate_semantic_source_paths(fragment, {"docs/design.md"})

    with pytest.raises(
        SemanticSchemaError,
        match=r"^semantic schema violation at nodes\[0\]\.source_file: source was not supplied$",
    ):
        validate_semantic_source_paths(fragment, {"docs/other.md"})


def test_runtime_prompt_schema_is_generated_from_the_canonical_owner():
    rendered = render_semantic_schema()
    assert rendered in llm._extraction_system()
    assert PROMPT_SCHEMA_VERSION in rendered
    assert "|".join(EDGE_RELATIONS) in rendered


# ---------- _call_claude_cli: argv shape ----------


def _make_envelope(result_obj: dict) -> str:
    return json.dumps(
        {
            "type": "result",
            "subtype": "success",
            "is_error": False,
            "result": json.dumps(result_obj),
            "usage": {
                "input_tokens": 1,
                "output_tokens": 1,
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 0,
            },
            "modelUsage": {"claude-opus-4-7": {}},
            "stop_reason": "end_turn",
        }
    )


def _prepare_claude_cli_mock(mock_run, result_obj: dict) -> None:
    _validated_executable.cache_clear()
    mock_run.side_effect = [
        MagicMock(
            returncode=0,
            stdout="\n".join(_REQUIRED_HELP_FLAGS),
            stderr="",
        ),
        MagicMock(
            returncode=0,
            stdout=_make_envelope(result_obj),
            stderr="",
        ),
    ]


@patch("shutil.which", return_value="/usr/local/bin/claude")
@patch("subprocess.run")
def test_instructions_ride_in_user_turn_with_hardened_system_prompt(mock_run, _which):
    """Extraction instructions stay in the user turn while the system prompt
    confines untrusted repository content.

    History: the original hollow-response cause was --append-system-prompt
    layering graphify's prompt on top of Claude Code's default agent prompt;
    the first fix switched to --system-prompt (replace). But newer Claude Code
    CLIs (>= ~2.1) don't treat --system-prompt as the sole authority — they
    keep the coding-agent context and reply conversationally to a bare file
    dump, which parses to zero nodes and gets bisected forever. The instructions
    now ride in the user turn (stdin); the system prompt is reserved for the
    capability boundary rather than extraction instructions."""
    _prepare_claude_cli_mock(mock_run, {"nodes": [], "edges": [], "hyperedges": []})
    llm._call_claude_cli("payload")
    argv = mock_run.call_args.args[0]
    assert "--system-prompt" in argv
    system_prompt = argv[argv.index("--system-prompt") + 1]
    assert "untrusted data" in system_prompt
    assert "Do not use tools" in system_prompt
    assert "--append-system-prompt" not in argv
    sent = mock_run.call_args.kwargs["input"]
    assert "graphify semantic extraction agent" in sent
    assert "output ONLY the JSON object" in sent
    assert "payload" in sent


@patch("shutil.which", return_value="/usr/local/bin/claude")
@patch("subprocess.run")
def test_model_env_var_adds_model_flag(mock_run, _which, monkeypatch):
    """GRAPHIFY_CLAUDE_CLI_MODEL must be forwarded to claude -p --model."""
    monkeypatch.setenv("GRAPHIFY_CLAUDE_CLI_MODEL", "haiku")
    _prepare_claude_cli_mock(mock_run, {"nodes": [], "edges": [], "hyperedges": []})
    llm._call_claude_cli("payload")
    argv = mock_run.call_args.args[0]
    assert "--model" in argv
    assert argv[argv.index("--model") + 1] == "haiku"


@patch("shutil.which", return_value="/usr/local/bin/claude")
@patch("subprocess.run")
def test_no_model_flag_when_env_var_unset(mock_run, _which, monkeypatch):
    """Default behaviour: when the env var is not set, --model is not
    added so claude-cli's own default kicks in."""
    monkeypatch.delenv("GRAPHIFY_CLAUDE_CLI_MODEL", raising=False)
    _prepare_claude_cli_mock(mock_run, {"nodes": [], "edges": [], "hyperedges": []})
    llm._call_claude_cli("payload")
    argv = mock_run.call_args.args[0]
    assert "--model" not in argv
