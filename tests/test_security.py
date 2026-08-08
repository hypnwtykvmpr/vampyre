"""Tests for graphify/security.py - URL validation, safe fetch, path guards, label sanitisation."""

from __future__ import annotations

import urllib.error
import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from graphify.security import (
    classify_bind_host,
    check_graph_file_size_cap,
    sanitize_label,
    sanitize_metadata,
    safe_fetch,
    safe_fetch_text,
    validate_graph_path,
    validate_url,
    _MAX_GRAPH_FILE_BYTES,
    _METADATA_MAX_LIST_ITEMS,
    _METADATA_MAX_VALUE_LEN,
    _sanitize_metadata_string,
    _sanitize_metadata_value,
)


def test_bind_host_classification_uses_every_resolved_address():
    def resolver(host, *_args, **_kwargs):
        addresses = {
            "loop.example": ["127.0.0.1", "::1"],
            "private.example": ["10.0.0.8"],
            "public.example": ["93.184.216.34"],
            "mixed.example": ["127.0.0.1", "93.184.216.34"],
        }[host]
        return [(2, 1, 6, "", (address, 8080)) for address in addresses]

    assert classify_bind_host("loop.example", 8080, resolver=resolver).kind == "loopback"
    assert classify_bind_host("private.example", 8080, resolver=resolver).kind == "private"
    assert classify_bind_host("public.example", 8080, resolver=resolver).kind == "public"
    assert classify_bind_host("mixed.example", 8080, resolver=resolver).kind == "mixed"
    assert classify_bind_host("0.0.0.0", 8080, resolver=resolver).kind == "wildcard"


def test_unresolved_bind_host_is_refused():
    def resolver(*_args, **_kwargs):
        raise OSError("dns unavailable")

    with pytest.raises(ValueError, match="could not be resolved"):
        classify_bind_host("missing.example", 8080, resolver=resolver)


def test_egress_endpoint_classification_uses_resolved_addresses():
    from graphify.egress import classify_endpoint

    def resolver(host, *_args, **_kwargs):
        addresses = {
            "loop.example": ["127.0.0.1", "::1"],
            "private.example": ["10.0.0.8"],
            "public.example": ["93.184.216.34"],
            "mixed.example": ["127.0.0.1", "93.184.216.34"],
        }[host]
        return [(2, 1, 6, "", (address, 443)) for address in addresses]

    assert classify_endpoint("https://loop.example", resolver=resolver) == "local_loopback"
    assert classify_endpoint("https://private.example", resolver=resolver) == "external_private"
    assert classify_endpoint("https://public.example", resolver=resolver) == "external_public"
    assert classify_endpoint("https://mixed.example", resolver=resolver) == "external_mixed"
    assert classify_endpoint("http://0.0.0.0:11434", resolver=resolver) == "external_wildcard"


def test_egress_endpoint_unresolved_is_not_misclassified_local():
    from graphify.egress import classify_endpoint

    def unresolved(*_args, **_kwargs):
        raise OSError("dns unavailable")

    assert classify_endpoint("https://model.invalid", resolver=unresolved) == "external_unresolved"


@pytest.mark.parametrize(
    ("endpoint", "endpoint_class", "eligible", "reason"),
    [
        ("http://127.0.0.1:11434/v1", "local_loopback", True, "eligible"),
        ("https://10.0.0.8/v1", "external_private", True, "eligible"),
        ("https://api.example/v1", "external_public", True, "eligible"),
        ("http://10.0.0.8/v1", "external_private", False, "plaintext_remote_endpoint"),
        ("http://api.example/v1", "external_public", False, "plaintext_remote_endpoint"),
        ("https://mixed.example/v1", "external_mixed", False, "unsafe_endpoint_class"),
        ("https://missing.example/v1", "external_unresolved", False, "unsafe_endpoint_class"),
        ("http://0.0.0.0:11434/v1", "external_wildcard", False, "unsafe_endpoint_class"),
    ],
)
def test_egress_decision_enforces_endpoint_transport_matrix(
    tmp_path, endpoint, endpoint_class, eligible, reason
):
    from graphify.egress import decide_egress

    source = tmp_path / "notes.md"
    decision = decide_egress(
        source,
        root=tmp_path,
        content=b"safe notes",
        backend="openai",
        endpoint=endpoint,
        endpoint_class=endpoint_class,
        model="model",
    )

    assert decision.eligible is eligible
    assert decision.reason_code == reason


@pytest.mark.parametrize(
    "path",
    [
        "config.secrets.yaml",
        "SECRETS/prod.md",
        "creds.json",
        "kubeconfig.yaml",
        ".env.production",
        "keys/service.pem",
    ],
)
def test_egress_high_confidence_credential_paths_are_blocked(path):
    from graphify.egress import credential_path_reason

    assert credential_path_reason(Path(path)) is not None


@pytest.mark.parametrize(
    "path",
    [
        "token.interceptor.ts",
        "password.validator.ts",
        "credentials.service.ts",
        "secret_handler.py",
        "docs/password-policy-discussion.md",
    ],
)
def test_egress_security_named_source_files_remain_eligible(path):
    from graphify.egress import credential_path_reason

    assert credential_path_reason(Path(path)) is None


@pytest.mark.parametrize(
    "content",
    [
        "AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE",
        "-----BEGIN PRIVATE KEY-----\nabc\n-----END PRIVATE KEY-----",
        'api_key = "sk-proj-abcdefghijklmnopqrstuvwxyz012345"',
        '"password": "h7Jx9Kp2Qw4Z"',
    ],
)
def test_egress_content_scanner_blocks_credential_material(content):
    from graphify.egress import credential_content_reason

    assert credential_content_reason(content.encode()) is not None


@pytest.mark.parametrize(
    "content",
    [
        "AWS_SECRET_ACCESS_KEY = 'Ab9xY2wV7qR4tN8mP3cK6dF1hJ5sL0zQ2uE7iO9p'",
        "SECRET_KEY = 'N8vQ3mZ7rT1kP6wX4cH9sD2yF5jL0bA7'",
        "db_password = 'Xq7#mR2$nL9!vB4@tK6'",
        "test_db_password = 'Xq7testM2nL9vB4tK6'",
    ],
)
def test_egress_content_scanner_blocks_compound_credential_assignments(content):
    from graphify.egress import credential_content_reason

    assert credential_content_reason(content.encode()) == "credential_assignment"


@pytest.mark.parametrize(
    "content",
    [
        "export class TokenInterceptor { intercept(request) { return request.token; } }",
        "password = user.password",
        "connect(user='neo4j', password='NEO4J_PASSWORD')",
        'const example = "replace-with-your-api-key";',
        "db_password = 'test-Password-X7!'",
        "AWS_SECRET_ACCESS_KEY = 'Ab9xY2wV7qR4tN8mP3cK6dEXAMPLE'",
        "SECRET_KEY = 'your-api-key-here'",
        "credential = os.environ['API_KEY']",
        "Use the X-Api-Key header for authentication.",
        "Document password validation and token rotation policy.",
    ],
)
def test_egress_content_scanner_keeps_noncredential_security_code(content):
    from graphify.egress import credential_content_reason

    assert credential_content_reason(content.encode()) is None


def test_egress_decision_contains_digest_but_never_source_content(tmp_path):
    from graphify.egress import decide_egress

    source = tmp_path / "notes.md"
    content = b"safe architecture notes"
    decision = decide_egress(
        source,
        root=tmp_path,
        content=content,
        backend="openai",
        endpoint="https://api.example.test/v1",
        model="test-model",
    )
    payload = decision.to_dict()
    assert payload["eligible"] is True
    assert payload["safe_relative_path"] == "notes.md"
    assert len(payload["content_digest"]) == 64
    assert len(payload["endpoint_digest"]) == 64
    assert "api.example.test" not in json.dumps(payload)
    assert "safe architecture notes" not in json.dumps(payload)


def test_outside_root_egress_decision_never_serializes_absolute_path(tmp_path):
    from graphify.egress import decide_egress

    outside = tmp_path.parent / "credential.txt"
    decision = decide_egress(
        outside,
        root=tmp_path,
        content=b"safe placeholder",
        backend="openai",
        endpoint="https://api.example.test/v1",
        model="test-model",
    )
    serialized = json.dumps(decision.to_dict())

    assert decision.eligible is False
    assert decision.safe_relative_path == "<outside-root>"
    assert str(outside) not in serialized


def test_egress_manifest_is_byte_stable_and_content_free(tmp_path):
    from graphify.egress import decide_egress, write_egress_manifest

    decisions = [
        decide_egress(
            tmp_path / name,
            root=tmp_path,
            content=content,
            backend="openai",
            endpoint="https://api.example.test/v1",
            model="test-model",
        )
        for name, content in (("b.md", b"safe b"), ("a.md", b"safe a"))
    ]
    output = tmp_path / "egress.json"
    write_egress_manifest(output, decisions)
    first = output.read_bytes()
    write_egress_manifest(output, list(reversed(decisions)))
    assert output.read_bytes() == first
    assert b"safe a" not in first and b"safe b" not in first
    assert [row["safe_relative_path"] for row in json.loads(first)["decisions"]] == [
        "a.md",
        "b.md",
    ]


def _credential_transform_variants(
    identifier: str, value: str, *, companion: str
) -> dict[str, str]:
    placeholder = f"replace-with-your-{identifier.casefold().replace('_', '-')}"
    return {
        "bare": f"{identifier} = {value}",
        "single-quoted": f"{identifier} = '{value}'",
        "double-quoted": f'{identifier} = "{value}"',
        "preceding-colon-line": f"Deployment settings:\n{identifier}: {value}",
        "nested-once": f"settings:\n  {identifier}: {value}",
        "nested-twice": f"settings:\n  auth:\n    {identifier}: {value}",
        "sequence-item": f"settings:\n  - {identifier}: {value}",
        "continuation": f"{identifier}:\n  {value}",
        "literal-block": f"{identifier}: |\n  {value}",
        "folded-block": f"{identifier}: >\n  {value}",
        "chomped-block": f"{identifier}: |-\n  {value}",
        "indented-block": f"{identifier}: |2\n    {value}",
        "trailing-comment": f"{identifier}: {value}  # provisioned",
        "crlf": f"settings:\r\n  {identifier}: {value}\r\n",
        "truncated-collection": f'"{identifier}": [\n  "{value}"',
        "benign-before-value": (f'"{identifier}": ["{placeholder}", "{companion}", "{value}"]'),
        "unclosed-triple": f'{identifier} = """\n{value}',
    }


_BOUNDARY_SECRETS = {
    "password": "P9mQ4vR2tN8xK5wC7sH3dF6yJ1bL0zA2",
    "api_key": "A8mQ4vR2tN9xK5wC7sH3dF6yJ1bL0zA3",
    "client_secret": "C7mQ4vR2tN9xK5wA8sH3dF6yJ1bL0zA4",
    "AWS_SECRET_ACCESS_KEY": "W6mQ4vR2tN9xK5wC8sH3dF7yJ1bL0zA5",
}
_BLOCKED_BOUNDARY_VARIANTS = [
    (identifier, shape, secret, content)
    for identifier, secret in _BOUNDARY_SECRETS.items()
    for shape, content in _credential_transform_variants(
        identifier,
        secret,
        companion="os.environ['API_KEY']",
    ).items()
]
_BENIGN_BOUNDARY_VARIANTS = [
    (
        identifier,
        shape,
        f"replace-with-your-{identifier.casefold().replace('_', '-')}",
        content,
    )
    for identifier in _BOUNDARY_SECRETS
    for shape, content in _credential_transform_variants(
        identifier,
        f"replace-with-your-{identifier.casefold().replace('_', '-')}",
        companion="os.environ['API_KEY']",
    ).items()
]


@pytest.mark.parametrize(
    ("identifier", "shape", "secret", "content"),
    _BLOCKED_BOUNDARY_VARIANTS,
    ids=[
        f"{identifier}-{shape}"
        for identifier, shape, _secret, _content in _BLOCKED_BOUNDARY_VARIANTS
    ],
)
def test_credential_transform_matrix_never_reaches_outbound_text(
    tmp_path, identifier, shape, secret, content
):
    from graphify import llm

    source = tmp_path / "docs" / "deployment.md"
    source.parent.mkdir(parents=True)
    source.write_text(content, encoding="utf-8")

    outbound, images, decisions = llm._prepare_egress_payloads(
        [source],
        root=tmp_path,
        backend="openai",
        cfg=llm.BACKENDS["openai"],
        model="test-model",
    )

    assert secret not in outbound, f"{identifier} leaked through {shape}"
    assert images == []
    assert len(decisions) == 1
    assert decisions[0]["eligible"] is False
    assert decisions[0]["reason_code"] == "credential_content"


@pytest.mark.parametrize(
    ("identifier", "shape", "placeholder", "content"),
    _BENIGN_BOUNDARY_VARIANTS,
    ids=[
        f"{identifier}-{shape}" for identifier, shape, _value, _content in _BENIGN_BOUNDARY_VARIANTS
    ],
)
def test_credential_transform_matrix_keeps_benign_controls_eligible(
    tmp_path, identifier, shape, placeholder, content
):
    from graphify import llm

    source = tmp_path / "docs" / "example.md"
    source.parent.mkdir(parents=True)
    source.write_text(content, encoding="utf-8")

    outbound, images, decisions = llm._prepare_egress_payloads(
        [source],
        root=tmp_path,
        backend="openai",
        cfg=llm.BACKENDS["openai"],
        model="test-model",
    )

    assert placeholder in outbound, f"{identifier} over-blocked {shape}"
    assert images == []
    assert len(decisions) == 1
    assert decisions[0]["eligible"] is True


@pytest.mark.parametrize(
    ("relative_path", "content"),
    [
        (
            "docs/deployment.md",
            "AWS_SECRET_ACCESS_KEY = 'Ab9xY2wV7qR4tN8mP3cK6dF1hJ5sL0zQ2uE7iO9p'",
        ),
        ("notes/onboarding.txt", "db_password = 'Xq7#mR2$nL9!vB4@tK6'"),
        ("deploy/values.yaml", "SECRET_KEY: 'N8vQ3mZ7rT1kP6wX4cH9sD2yF5jL0bA7'"),
        (
            "docs/colon-prefixed.md",
            "Set this:\nSECRET_KEY = 'K9rQ4mZ7tV2xP6wN8cH3sD5yF1jL0bA4'",
        ),
        (
            "deploy/nested-values.yaml",
            "env:\n  AWS_SECRET_ACCESS_KEY: 'Q7mN2vK9rT4xP6wC8sH3dF5yJ1bL0zA2'",
        ),
        (
            "deploy/literal-values.yaml",
            "password: |\n  R8mQ3vN7tK2xP5wC9sH4dF6yJ1bL0zA3",
        ),
        (
            "deploy/folded-values.yaml",
            "secret_token: >\n  T6nR2vM9qK4xP7wC8sH3dF5yJ1bL0zA4",
        ),
        (
            "deploy/explained-literal.yaml",
            "password: |\n  Provision with R5mQ8vN2tK7xP4wC9sH3dF6yJ1bL0zA2",
        ),
        (
            "deploy/quoted-values.yaml",
            "password: 'Xq7!O''Brien-R2nL9vB4tK6'",
        ),
        (
            "deploy/plain-multiline.yaml",
            "password:\n    V9mQ2rN7tK4xP6wC8sH3dF5yJ1bL0zA2",
        ),
        (
            "deploy/sequence-values.yaml",
            "api_key:\n    - Q8nR3vM7tK2xP5wC9sH4dF6yJ1bL0zA3",
        ),
        (
            "docs/pretty-json.md",
            '"client_secret":\n    "T7mR2vN9qK4xP6wC8sH3dF5yJ1bL0zA4"',
        ),
        (
            "docs/toml-config.md",
            'secret = """\nR6mQ3vN8tK2xP5wC9sH4dF7yJ1bL0zA5\n"""',
        ),
        (
            "deploy/commented-value.yaml",
            "password: # provisioned below\n  P5mQ8vN2tK7xR4wC9sH3dF6yJ1bL0zA7",
        ),
        (
            "deploy/folded-plain-value.yaml",
            "password: Provisioned with\n  N4mQ7vR2tK8xP5wC9sH3dF6yJ1bL0zA8",
        ),
        (
            "docs/bracketed-json.md",
            '"api_key": [\n  "M3mQ7vR2tK8xP5wC9sH4dF6yJ1bL0zA9"\n]',
        ),
        (
            "docs/literal-toml.md",
            "secret = '''\nL2mQ7vR3tK8xP5wC9sH4dF6yJ1bL0zA0\n'''",
        ),
        (
            "docs/plain-value.md",
            "password:\nK1mQ7vR3tN8xP5wC9sH4dF6yJ2bL0zA1",
        ),
        (
            "docs/flat-array.md",
            '"api_key": [\n"J9mQ7vR3tN8xP5wC2sH4dF6yK1bL0zA2"\n]',
        ),
        (
            "docs/hcl-heredoc.md",
            "password = <<EOF\nH8mQ7vR3tN9xP5wC2sK4dF6yJ1bL0zA3\nEOF",
        ),
        (
            "docs/colon-value.md",
            "password:\nG7mQ4vR2tN9x:P5wC8sK3dF6yJ1bL0zA4",
        ),
        (
            "deploy/hash-prefixed-value.yaml",
            "password: |\n  #F6mQ4vR2tN9xP5wC8sK3dH7yJ1bL0zA5",
        ),
        (
            "docs/nested-json.md",
            '"api_key": [\n  ["replace-with-your-api-key"],\n'
            '  "E5mQ4vR2tN9xP6wC8sK3dH7yJ1bL0zA6"\n]',
        ),
        (
            "docs/inline-array.md",
            '"api_key": ["replace-with-your-api-key", "D4mQ7vR2tN9xP5wC8sK3dH6yJ1bL0zA7"]',
        ),
        (
            "docs/nested-object.md",
            '"client_secret": {"example": "your-client-secret", '
            '"active": "C3mQ7vR2tN9xP5wK8sH4dF6yJ1bL0zA8"}',
        ),
        (
            "docs/truncated-array.md",
            '"api_key": [\n  "B2mQ7vR3tN9xP5wK8sH4dF6yJ1bL0zA9"',
        ),
    ],
)
def test_blocked_source_makes_zero_provider_calls(tmp_path, monkeypatch, relative_path, content):
    from graphify import llm

    secret = tmp_path / relative_path
    secret.parent.mkdir(parents=True, exist_ok=True)
    secret.write_text(content, encoding="utf-8")
    monkeypatch.setattr(
        llm,
        "_call_openai_compat",
        lambda *_args, **_kwargs: pytest.fail("blocked content reached provider"),
    )
    result = llm.extract_files_direct([secret], backend="openai", api_key="fake", root=tmp_path)
    assert result["nodes"] == []
    assert result["egress_decisions"][0]["eligible"] is False
    assert result["egress_decisions"][0]["reason_code"] == "credential_content"
    assert result["egress_decisions"][0]["safe_relative_path"] == relative_path


def test_yaml_block_scalar_placeholder_remains_eligible(tmp_path, monkeypatch):
    from graphify import llm

    source = tmp_path / "deploy" / "example.yaml"
    source.parent.mkdir(parents=True)
    source.write_text(
        "password: |\n  Replace with your-password-here before deployment.\n",
        encoding="utf-8",
    )
    captured = {}

    def provider(_url, _key, _model, user_message, **_kwargs):
        captured["user_message"] = user_message
        return {"nodes": [], "edges": [], "hyperedges": []}

    monkeypatch.setattr(llm, "_call_openai_compat", provider)
    result = llm.extract_files_direct([source], backend="openai", api_key="fake", root=tmp_path)

    assert "your-password-here" in captured["user_message"]
    assert result["egress_decisions"][0]["eligible"] is True


@pytest.mark.parametrize(
    "content",
    [
        "password:\n  your-password-here",
        "api_key:\n  - os.environ['API_KEY']",
        '"client_secret":\n  "replace-with-your-client-secret"',
        'secret = """\nExample placeholder only\n"""',
        "password: Provision through\n  the deployment environment",
        "password:\nyour-password-here",
        '"api_key": [\n"os.environ[\'API_KEY\']"\n]',
        '"api_key": [\n  ["replace-with-your-api-key"],\n  ["os.environ[\'API_KEY\']"]\n]',
        '"api_key": ["replace-with-your-api-key", "os.environ[\'API_KEY\']"]',
        '"client_secret": {"example": "your-client-secret"}',
        '"api_key": [\n  "replace-with-your-api-key"',
        "password = <<EOF\nreplace-with-your-password\nEOF",
    ],
)
def test_multiline_credential_placeholders_remain_eligible(tmp_path, monkeypatch, content):
    from graphify import llm

    source = tmp_path / "docs" / "example.md"
    source.parent.mkdir(parents=True)
    source.write_text(content, encoding="utf-8")
    captured = {}

    def provider(_url, _key, _model, user_message, **_kwargs):
        captured["user_message"] = user_message
        return {"nodes": [], "edges": [], "hyperedges": []}

    monkeypatch.setattr(llm, "_call_openai_compat", provider)
    result = llm.extract_files_direct([source], backend="openai", api_key="fake", root=tmp_path)

    assert content in captured["user_message"]
    assert result["egress_decisions"][0]["eligible"] is True


def test_security_topic_document_is_content_scanned_before_provider_egress(tmp_path, monkeypatch):
    from graphify import llm

    source = tmp_path / "secret_handler.txt"
    source.write_text("How the credential handler validates inputs.", encoding="utf-8")
    captured = {}

    def provider(_url, _key, _model, user_message, **_kwargs):
        captured["user_message"] = user_message
        return {"nodes": [], "edges": [], "hyperedges": []}

    monkeypatch.setattr(llm, "_call_openai_compat", provider)
    clean = llm.extract_files_direct([source], backend="openai", api_key="fake", root=tmp_path)
    assert "credential handler" in captured["user_message"]
    assert clean["egress_decisions"][0]["eligible"] is True

    source.write_text(
        "db_password = 'Xq7#mR2$nL9!vB4@tK6'",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        llm,
        "_call_openai_compat",
        lambda *_args, **_kwargs: pytest.fail("credential content reached provider"),
    )
    blocked = llm.extract_files_direct([source], backend="openai", api_key="fake", root=tmp_path)
    assert blocked["nodes"] == []
    assert blocked["egress_decisions"][0]["eligible"] is False
    assert blocked["egress_decisions"][0]["reason_code"] == "credential_content"


def test_mixed_chunk_sends_safe_bytes_only(tmp_path, monkeypatch):
    from graphify import llm

    safe = tmp_path / "architecture.md"
    blocked = tmp_path / "private.md"
    safe.write_text("SAFE ARCHITECTURE", encoding="utf-8")
    blocked.write_text("OPENAI_API_KEY=sk-proj-abcdefghijklmnopqrstuvwxyz012345", encoding="utf-8")
    captured = {}

    def provider(_url, _key, _model, user_message, **_kwargs):
        captured["user_message"] = user_message
        return {"nodes": [], "edges": [], "hyperedges": []}

    monkeypatch.setattr(llm, "_call_openai_compat", provider)
    result = llm.extract_files_direct(
        [blocked, safe], backend="openai", api_key="fake", root=tmp_path
    )
    assert "SAFE ARCHITECTURE" in captured["user_message"]
    assert "sk-proj" not in captured["user_message"]
    assert sum(not row["eligible"] for row in result["egress_decisions"]) == 1


def test_credential_bytes_embedded_in_image_make_zero_provider_calls(tmp_path, monkeypatch):
    from graphify import llm

    image = tmp_path / "screenshot.png"
    image.write_bytes(b"PNG\x00AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE")
    monkeypatch.setattr(
        llm,
        "_call_openai_compat",
        lambda *_args, **_kwargs: pytest.fail("blocked image reached provider"),
    )
    result = llm.extract_files_direct([image], backend="openai", api_key="fake", root=tmp_path)
    assert result["nodes"] == []
    assert result["egress_decisions"][0]["reason_code"] == "credential_content"


def test_parallel_extraction_persists_stable_egress_manifest(tmp_path, monkeypatch):
    from graphify import llm

    safe = tmp_path / "architecture.md"
    blocked = tmp_path / "private.md"
    path_blocked = tmp_path / "config.secrets.yaml"
    safe.write_text("SAFE ARCHITECTURE", encoding="utf-8")
    blocked.write_text("OPENAI_API_KEY=sk-proj-abcdefghijklmnopqrstuvwxyz012345", encoding="utf-8")
    path_blocked.write_text("password: h7Jx9Kp2Qw4Z", encoding="utf-8")
    manifest = tmp_path / "out" / "egress-manifest.json"
    monkeypatch.setattr(
        llm,
        "_call_openai_compat",
        lambda *_args, **_kwargs: {
            "nodes": [],
            "edges": [],
            "hyperedges": [],
            "input_tokens": 0,
            "output_tokens": 0,
        },
    )

    first_result = llm.extract_corpus_parallel(
        [blocked, safe],
        backend="openai",
        api_key="fake",
        root=tmp_path,
        max_concurrency=1,
        egress_manifest_path=manifest,
        preblocked_files=[path_blocked],
    )
    first_bytes = manifest.read_bytes()
    second_result = llm.extract_corpus_parallel(
        [safe, blocked],
        backend="openai",
        api_key="fake",
        root=tmp_path,
        max_concurrency=1,
        egress_manifest_path=manifest,
        preblocked_files=[path_blocked],
    )

    assert manifest.read_bytes() == first_bytes
    assert len(first_result["egress_decisions"]) == 3
    assert len(second_result["egress_decisions"]) == 3
    assert first_result["semantic_provenance"] == second_result["semantic_provenance"]
    assert first_result["semantic_provenance"]["backend"] == "openai"
    assert len(first_result["semantic_provenance"]["endpoint_digest"]) == 64
    assert b"SAFE ARCHITECTURE" not in first_bytes
    assert b"sk-proj" not in first_bytes
    assert b"h7Jx9Kp2Qw4Z" not in first_bytes


def test_parallel_merge_revalidates_every_semantic_source_path(tmp_path, monkeypatch):
    from graphify import llm
    from graphify.semantic_schema import SemanticSchemaError

    source = tmp_path / "architecture.md"
    source.write_text("SAFE ARCHITECTURE", encoding="utf-8")
    monkeypatch.setattr(
        llm,
        "_extract_with_adaptive_retry",
        lambda *_args, **_kwargs: {
            "nodes": [
                {
                    "id": "fabricated",
                    "label": "Fabricated",
                    "file_type": "document",
                    "source_file": "not-supplied.md",
                }
            ],
            "edges": [],
            "hyperedges": [],
            "input_tokens": 0,
            "output_tokens": 0,
            "complete": True,
            "egress_decisions": [
                {
                    "eligible": True,
                    "safe_relative_path": "architecture.md",
                }
            ],
        },
    )

    with pytest.raises(SemanticSchemaError, match="source was not supplied"):
        llm.extract_corpus_parallel(
            [source],
            backend="openai",
            api_key="fake",
            root=tmp_path,
            max_concurrency=1,
        )


def test_preblocked_egress_hashes_without_loading_entire_file(tmp_path, monkeypatch):
    from graphify import llm

    blocked = tmp_path / "config.secrets.yaml"
    blocked.write_bytes(b"x" * (2 * 1024 * 1024))
    monkeypatch.setattr(
        Path,
        "read_bytes",
        lambda _self: pytest.fail("preblocked files must be streamed, not loaded wholesale"),
    )

    result = llm.extract_corpus_parallel(
        [],
        backend="openai",
        api_key="fake",
        root=tmp_path,
        max_concurrency=1,
        preblocked_files=[blocked],
    )

    assert result["egress_decisions"][0]["eligible"] is False
    assert len(result["egress_decisions"][0]["content_digest"]) == 64


# ---------------------------------------------------------------------------
# validate_url
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _stable_public_dns(monkeypatch):
    """Keep URL validation offline while preserving public-address checks."""

    def resolver(*_args, **_kwargs):
        return [(2, 1, 6, "", ("93.184.216.34", 0))]

    monkeypatch.setattr("graphify.security.socket.getaddrinfo", resolver)
    monkeypatch.setattr("graphify.egress.socket.getaddrinfo", resolver)


def test_validate_url_accepts_http():
    assert validate_url("http://example.com/page") == "http://example.com/page"


def test_validate_url_accepts_https():
    assert validate_url("https://arxiv.org/abs/1706.03762") == "https://arxiv.org/abs/1706.03762"


def test_validate_url_rejects_file():
    with pytest.raises(ValueError, match="file"):
        validate_url("file:///etc/passwd")


def test_validate_url_rejects_ftp():
    with pytest.raises(ValueError, match="ftp"):
        validate_url("ftp://files.example.com/data.zip")


def test_validate_url_rejects_data():
    with pytest.raises(ValueError, match="data"):
        validate_url("data:text/html,<script>alert(1)</script>")


def test_validate_url_rejects_empty_scheme():
    with pytest.raises(ValueError):
        validate_url("//no-scheme.example.com")


# ---------------------------------------------------------------------------
# safe_fetch - scheme and redirect guards (mocked network)
# ---------------------------------------------------------------------------


def _make_mock_response(content: bytes, status: int = 200):
    mock = MagicMock()
    mock.__enter__ = lambda s: s
    mock.__exit__ = MagicMock(return_value=False)
    mock.status = status
    mock.code = status
    chunks = [content[i : i + 65536] for i in range(0, len(content), 65536)] + [b""]
    mock.read.side_effect = chunks
    return mock


def test_safe_fetch_rejects_file_url():
    with pytest.raises(ValueError, match="file"):
        safe_fetch("file:///etc/passwd")


def test_safe_fetch_rejects_ftp_url():
    with pytest.raises(ValueError, match="ftp"):
        safe_fetch("ftp://example.com/file.zip")


def test_safe_fetch_returns_bytes(tmp_path):
    mock_resp = _make_mock_response(b"hello world")
    with patch("graphify.security._build_opener") as mock_opener_fn:
        mock_opener = MagicMock()
        mock_opener.open.return_value = mock_resp
        mock_opener_fn.return_value = mock_opener
        result = safe_fetch("https://example.com/")
    assert result == b"hello world"


def test_safe_fetch_raises_on_non_2xx():
    mock_resp = _make_mock_response(b"Not Found", status=404)
    with patch("graphify.security._build_opener") as mock_opener_fn:
        mock_opener = MagicMock()
        mock_opener.open.return_value = mock_resp
        mock_opener_fn.return_value = mock_opener
        with pytest.raises(urllib.error.HTTPError):
            safe_fetch("https://example.com/missing")


def test_safe_fetch_raises_on_size_exceeded():
    # Build a response larger than max_bytes
    big_chunk = b"x" * 65_537
    mock_resp = MagicMock()
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)
    mock_resp.status = 200
    mock_resp.code = 200
    # Return the chunk twice so total > max_bytes=65536
    mock_resp.read.side_effect = [big_chunk, big_chunk, b""]

    with patch("graphify.security._build_opener") as mock_opener_fn:
        mock_opener = MagicMock()
        mock_opener.open.return_value = mock_resp
        mock_opener_fn.return_value = mock_opener
        with pytest.raises(OSError, match="size limit"):
            safe_fetch("https://example.com/huge", max_bytes=65_536)


# ---------------------------------------------------------------------------
# safe_fetch_text
# ---------------------------------------------------------------------------


def test_safe_fetch_text_decodes_utf8():
    content = "héllo wörld".encode("utf-8")
    mock_resp = _make_mock_response(content)
    with patch("graphify.security._build_opener") as mock_opener_fn:
        mock_opener = MagicMock()
        mock_opener.open.return_value = mock_resp
        mock_opener_fn.return_value = mock_opener
        result = safe_fetch_text("https://example.com/")
    assert result == "héllo wörld"


def test_safe_fetch_text_replaces_bad_bytes():
    bad = b"hello \xff world"
    mock_resp = _make_mock_response(bad)
    with patch("graphify.security._build_opener") as mock_opener_fn:
        mock_opener = MagicMock()
        mock_opener.open.return_value = mock_resp
        mock_opener_fn.return_value = mock_opener
        result = safe_fetch_text("https://example.com/")
    assert "hello" in result
    assert "world" in result
    assert "\xff" not in result


# ---------------------------------------------------------------------------
# validate_graph_path
# ---------------------------------------------------------------------------


def test_validate_graph_path_allows_inside_base(tmp_path):
    base = tmp_path / "graphify-out"
    base.mkdir()
    graph = base / "graph.json"
    graph.write_text("{}", encoding="utf-8")
    result = validate_graph_path(str(graph), base=base)
    assert result == graph.resolve()


def test_validate_graph_path_blocks_traversal(tmp_path):
    base = tmp_path / "graphify-out"
    base.mkdir()
    evil = tmp_path / "graphify-out" / ".." / "etc_passwd"
    with pytest.raises(ValueError, match="escapes"):
        validate_graph_path(str(evil), base=base)


def test_validate_graph_path_requires_base_exists(tmp_path):
    base = tmp_path / "graphify-out"  # not created
    with pytest.raises(ValueError, match="does not exist"):
        validate_graph_path(str(base / "graph.json"), base=base)


def test_validate_graph_path_raises_if_file_missing(tmp_path):
    base = tmp_path / "graphify-out"
    base.mkdir()
    with pytest.raises(FileNotFoundError):
        validate_graph_path(str(base / "missing.json"), base=base)


def test_validate_graph_path_default_base_discovers_output_dir(tmp_path):
    """With base omitted, the output dir is discovered by walking the path's
    parents for the configured output-dir name (default 'graphify-out')."""
    base = tmp_path / "graphify-out"
    base.mkdir()
    graph = base / "graph.json"
    graph.write_text("{}", encoding="utf-8")
    assert validate_graph_path(str(graph)) == graph.resolve()


def test_validate_graph_path_default_base_honours_graphify_out_override(tmp_path, monkeypatch):
    """The base=None discovery must honour GRAPHIFY_OUT, not the hardcoded
    'graphify-out' literal — otherwise a renamed output dir validates against the
    wrong base or raises spuriously (#1423)."""
    monkeypatch.setattr("graphify.security.GRAPHIFY_OUT_NAME", "custom-out")
    monkeypatch.setattr("graphify.security.GRAPHIFY_OUT", "custom-out")
    out = tmp_path / "custom-out"
    out.mkdir()
    graph = out / "graph.json"
    graph.write_text("{}", encoding="utf-8")
    # No base passed → must discover custom-out by name rather than graphify-out.
    assert validate_graph_path(str(graph)) == graph.resolve()


# ---------------------------------------------------------------------------
# sanitize_label
# ---------------------------------------------------------------------------


def test_sanitize_label_passthrough_html_chars():
    # sanitize_label does NOT HTML-escape — callers that inject into HTML must
    # wrap with html.escape() themselves (e.g. the title in to_html())
    assert sanitize_label("<script>") == "<script>"
    assert sanitize_label("foo & bar") == "foo & bar"


def test_sanitize_label_strips_control_chars():
    result = sanitize_label("hello\x00\x1fworld")
    assert "\x00" not in result
    assert "\x1f" not in result
    assert "helloworld" in result


def test_sanitize_label_caps_at_256():
    long_label = "a" * 300
    assert len(sanitize_label(long_label)) <= 256


def test_sanitize_label_safe_passthrough():
    assert sanitize_label("MyClass") == "MyClass"
    assert sanitize_label("extract_python") == "extract_python"


# ---------------------------------------------------------------------------
# check_graph_file_size_cap (#F4 — graph-load memory bomb protection)
# ---------------------------------------------------------------------------


def test_graph_size_cap_default_is_512_mib():
    assert _MAX_GRAPH_FILE_BYTES == 512 * 1024 * 1024


def test_graph_size_cap_under_limit_returns_none(tmp_path):
    p = tmp_path / "graph.json"
    p.write_text('{"nodes": [], "links": []}', encoding="utf-8")
    assert check_graph_file_size_cap(p) is None


def test_graph_size_cap_over_limit_raises(monkeypatch, tmp_path):
    monkeypatch.setattr("graphify.security._MAX_GRAPH_FILE_BYTES", 16)
    p = tmp_path / "graph.json"
    p.write_text('{"nodes": [], "links": [], "padding": "x" * 50}', encoding="utf-8")
    with pytest.raises(ValueError, match="exceeds"):
        check_graph_file_size_cap(p)


def test_graph_size_cap_error_message_includes_size_and_cap(monkeypatch, tmp_path):
    monkeypatch.setattr("graphify.security._MAX_GRAPH_FILE_BYTES", 8)
    p = tmp_path / "graph.json"
    p.write_text("AAAAAAAAAAAAAAAA", encoding="utf-8")  # 16 bytes
    with pytest.raises(ValueError) as excinfo:
        check_graph_file_size_cap(p)
    msg = str(excinfo.value)
    assert "16" in msg  # observed size
    assert "8" in msg  # cap
    assert "byte" in msg.lower()


def test_graph_size_cap_at_boundary_passes(monkeypatch, tmp_path):
    # Boundary: equal to cap is allowed; strictly greater is rejected.
    p = tmp_path / "graph.json"
    payload = "A" * 32
    p.write_text(payload, encoding="utf-8")
    monkeypatch.setattr("graphify.security._MAX_GRAPH_FILE_BYTES", 32)
    assert check_graph_file_size_cap(p) is None
    monkeypatch.setattr("graphify.security._MAX_GRAPH_FILE_BYTES", 31)
    with pytest.raises(ValueError):
        check_graph_file_size_cap(p)


def test_graph_size_cap_missing_file_silently_returns(tmp_path):
    # When stat() fails (FileNotFoundError → OSError), the helper returns None
    # so the caller's own existence check can surface a clearer error.
    missing = tmp_path / "does_not_exist.json"
    assert check_graph_file_size_cap(missing) is None


def test_graph_size_cap_unreadable_directory_silently_returns(monkeypatch, tmp_path):
    # Force stat() to raise PermissionError → still OSError → silent return.
    p = tmp_path / "graph.json"
    p.write_text("{}", encoding="utf-8")

    def _boom(self):
        raise PermissionError("denied")

    monkeypatch.setattr(Path, "stat", _boom)
    assert check_graph_file_size_cap(p) is None


# ---------------------------------------------------------------------------
# sanitize_metadata (recursive, bounded, HTML-safe)
# ---------------------------------------------------------------------------


def test_sanitize_metadata_string_strips_control_chars():
    result = _sanitize_metadata_string("hello\x00\x1fworld")
    assert "\x00" not in result
    assert "\x1f" not in result
    assert "helloworld" in result


def test_sanitize_metadata_string_escapes_html():
    result = _sanitize_metadata_string("<script>alert('x')</script>")
    assert "&lt;" in result
    assert "&gt;" in result
    assert "<script>" not in result


def test_sanitize_metadata_string_escapes_quotes():
    result = _sanitize_metadata_string("a\"b'c")
    # quote=True escapes both " and '
    assert "&quot;" in result
    assert "&#x27;" in result or "&apos;" in result


def test_sanitize_metadata_string_caps_length():
    long = "a" * (_METADATA_MAX_VALUE_LEN + 100)
    result = _sanitize_metadata_string(long)
    assert len(result) <= _METADATA_MAX_VALUE_LEN


def test_sanitize_metadata_string_coerces_non_string():
    # Non-str/dict/list/scalar inputs route through string sanitisation.
    class _Custom:
        def __str__(self) -> str:
            return "custom-repr"

    assert _sanitize_metadata_string(_Custom()) == "custom-repr"


def test_sanitize_metadata_value_preserves_simple_types():
    assert _sanitize_metadata_value(42) == 42
    assert _sanitize_metadata_value(3.14) == 3.14
    assert _sanitize_metadata_value(True) is True
    assert _sanitize_metadata_value(False) is False
    assert _sanitize_metadata_value(None) is None


def test_sanitize_metadata_value_recurses_into_dict():
    out = _sanitize_metadata_value({"k": "<script>x</script>"})
    assert isinstance(out, dict)
    assert "&lt;" in out["k"]


def test_sanitize_metadata_value_recurses_into_list():
    out = _sanitize_metadata_value(["<a>", "<b>", "<c>"])
    assert isinstance(out, list)
    assert all("&lt;" in s for s in out)


def test_sanitize_metadata_value_caps_list_length():
    huge = list(range(_METADATA_MAX_LIST_ITEMS * 3))
    out = _sanitize_metadata_value(huge)
    assert isinstance(out, list)
    assert len(out) == _METADATA_MAX_LIST_ITEMS


def test_sanitize_metadata_value_converts_tuple_to_list():
    out = _sanitize_metadata_value(("a", "b"))
    assert isinstance(out, list)
    assert out == ["a", "b"]


def test_sanitize_metadata_none_returns_empty_dict():
    assert sanitize_metadata(None) == {}


def test_sanitize_metadata_drops_empty_key():
    # Empty key (after control-char strip) is dropped.
    out = sanitize_metadata({"\x00": "v", "k": "v2"})
    assert "\x00" not in out
    assert out.get("k") == "v2"
    assert len(out) == 1


def test_sanitize_metadata_sanitizes_keys():
    out = sanitize_metadata({"<bad>": "v"})
    assert "<bad>" not in out
    assert any("&lt;" in k for k in out.keys())


def test_sanitize_metadata_recursive_nested():
    raw: dict[str, Any] = {
        "outer": {
            "inner": "<script>x</script>",
            "list": ["a", "<b>", 99, None, True],
        },
        "scalar": 42,
    }
    out = sanitize_metadata(raw)
    assert isinstance(out["outer"], dict)
    inner = out["outer"]
    assert isinstance(inner, dict)
    assert "&lt;" in inner["inner"]
    items = inner["list"]
    assert isinstance(items, list)
    assert items[0] == "a"
    assert "&lt;" in items[1]
    assert items[2] == 99
    assert items[3] is None
    assert items[4] is True
    assert out["scalar"] == 42


def test_sanitize_metadata_bool_not_coerced_to_int():
    # bool is an int subclass — order of isinstance checks must preserve bool.
    out = sanitize_metadata({"flag_t": True, "flag_f": False, "num": 1})
    assert out["flag_t"] is True
    assert out["flag_f"] is False
    assert out["num"] == 1
