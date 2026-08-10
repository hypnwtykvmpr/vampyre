"""Tests for the Streamable HTTP transport on the MCP server (issue #1143).

These exercise the ASGI wiring in-process (no uvicorn, no real socket) via
Starlette's TestClient, so they stay fast and offline. The stdio path is
unchanged and covered elsewhere.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import mcp  # noqa: F401
import starlette  # noqa: F401

from starlette.testclient import TestClient  # noqa: E402
from mcp import ClientSession  # noqa: E402
from mcp.client.stdio import StdioServerParameters, stdio_client  # noqa: E402
from mcp.types import TextContent  # noqa: E402

from graphify import serve as serve_mod  # noqa: E402

SAMPLE_GRAPH = {
    "directed": True,
    "nodes": [
        {"id": "a", "label": "Alpha", "community": 0},
        {"id": "b", "label": "Beta", "community": 0},
    ],
    "edges": [
        {"source": "a", "target": "b", "relation": "calls", "confidence": "EXTRACTED"},
    ],
}

_INIT_BODY = {
    "jsonrpc": "2.0",
    "id": 1,
    "method": "initialize",
    "params": {
        "protocolVersion": "2025-03-26",
        "capabilities": {},
        "clientInfo": {"name": "test", "version": "0"},
    },
}

_MCP_HEADERS = {
    "Content-Type": "application/json",
    "Accept": "application/json, text/event-stream",
}


def test_http_docstring_names_supported_authentication_headers() -> None:
    guidance = serve_mod.serve_http.__doc__ or ""

    assert "Clients authenticate with either" in guidance
    assert "``Authorization: Bearer <key>``" in guidance
    assert "``X-API-Key: <key>``" in guidance


def _graph_file(tmp_path: Path) -> str:
    p = tmp_path / "graph.json"
    p.write_text(json.dumps(SAMPLE_GRAPH), encoding="utf-8")
    return str(p)


@pytest.mark.parametrize(
    "host,expected",
    [("", True), ("0.0.0.0", True), ("::", True), ("127.0.0.1", False), ("localhost", False)],
)
def test_wildcard_bind_host_detection(host, expected):
    assert serve_mod._is_wildcard_bind_host(host) is expected


def _client(app) -> TestClient:
    # Default host is 127.0.0.1, so the DNS-rebinding guard only accepts that
    # Host header (TestClient otherwise sends the disallowed "testserver").
    return TestClient(app, base_url="http://127.0.0.1")


def test_app_builds_and_initialize_succeeds(tmp_path):
    app = serve_mod._build_http_app(_graph_file(tmp_path), json_response=True)
    with _client(app) as client:
        resp = client.post("/mcp", headers=_MCP_HEADERS, json=_INIT_BODY)
        assert resp.status_code == 200
        # json_response=True returns a single JSON-RPC envelope.
        payload = resp.json()
        assert payload["jsonrpc"] == "2.0"
        assert payload["result"]["serverInfo"]["name"] == "graphify"


def test_unknown_path_is_404(tmp_path):
    app = serve_mod._build_http_app(_graph_file(tmp_path), json_response=True)
    with _client(app) as client:
        resp = client.post("/nope", headers=_MCP_HEADERS, json=_INIT_BODY)
        assert resp.status_code == 404


def test_api_key_missing_is_401(tmp_path):
    app = serve_mod._build_http_app(_graph_file(tmp_path), api_key="s3cret", json_response=True)
    with _client(app) as client:
        resp = client.post("/mcp", headers=_MCP_HEADERS, json=_INIT_BODY)
        assert resp.status_code == 401
        assert resp.json()["error"] == "unauthorized"


def test_api_key_wrong_is_401(tmp_path):
    app = serve_mod._build_http_app(_graph_file(tmp_path), api_key="s3cret", json_response=True)
    with _client(app) as client:
        resp = client.post(
            "/mcp",
            headers={**_MCP_HEADERS, "Authorization": "Bearer nope"},
            json=_INIT_BODY,
        )
        assert resp.status_code == 401


def test_api_key_bearer_ok(tmp_path):
    app = serve_mod._build_http_app(_graph_file(tmp_path), api_key="s3cret", json_response=True)
    with _client(app) as client:
        resp = client.post(
            "/mcp",
            headers={**_MCP_HEADERS, "Authorization": "Bearer s3cret"},
            json=_INIT_BODY,
        )
        assert resp.status_code == 200
        assert resp.json()["result"]["serverInfo"]["name"] == "graphify"


def test_api_key_x_api_key_header_ok(tmp_path):
    app = serve_mod._build_http_app(_graph_file(tmp_path), api_key="s3cret", json_response=True)
    with _client(app) as client:
        resp = client.post(
            "/mcp",
            headers={**_MCP_HEADERS, "X-API-Key": "s3cret"},
            json=_INIT_BODY,
        )
        assert resp.status_code == 200


def test_blank_api_key_means_no_auth(tmp_path):
    # An empty/whitespace key must normalize to "no auth", not a key of "".
    app = serve_mod._build_http_app(_graph_file(tmp_path), api_key="   ", json_response=True)
    with _client(app) as client:
        resp = client.post("/mcp", headers=_MCP_HEADERS, json=_INIT_BODY)
        assert resp.status_code == 200


def test_api_key_bearer_scheme_case_insensitive(tmp_path):
    app = serve_mod._build_http_app(_graph_file(tmp_path), api_key="s3cret", json_response=True)
    with _client(app) as client:
        resp = client.post(
            "/mcp",
            headers={**_MCP_HEADERS, "Authorization": "bearer s3cret"},
            json=_INIT_BODY,
        )
        assert resp.status_code == 200


def test_custom_mount_path(tmp_path):
    app = serve_mod._build_http_app(_graph_file(tmp_path), path="/graph", json_response=True)
    with _client(app) as client:
        ok = client.post("/graph", headers=_MCP_HEADERS, json=_INIT_BODY)
        assert ok.status_code == 200
        missing = client.post("/mcp", headers=_MCP_HEADERS, json=_INIT_BODY)
        assert missing.status_code == 404


def test_tools_list_over_http(tmp_path):
    """A full initialize -> tools/list round trip works over the HTTP transport."""
    app = serve_mod._build_http_app(_graph_file(tmp_path), json_response=True)
    with _client(app) as client:
        init = client.post("/mcp", headers=_MCP_HEADERS, json=_INIT_BODY)
        assert init.status_code == 200
        session_id = init.headers.get("mcp-session-id")
        assert session_id, "stateful transport should return a session id"
        notify_headers = {**_MCP_HEADERS, "mcp-session-id": session_id}
        client.post(
            "/mcp",
            headers=notify_headers,
            json={"jsonrpc": "2.0", "method": "notifications/initialized"},
        )
        resp = client.post(
            "/mcp",
            headers=notify_headers,
            json={"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
        )
        assert resp.status_code == 200
        names = {t["name"] for t in resp.json()["result"]["tools"]}
        assert {"query_graph", "get_node", "graph_stats"} <= names


@pytest.mark.parametrize(
    "arguments",
    [
        {"question": "x" * 8193},
        {"question": "alpha", "depth": 0},
        {"question": "alpha", "token_budget": 0},
        {"question": "alpha", "context_filter": ["x" * 129]},
    ],
)
def test_query_graph_refuses_unbounded_arguments(tmp_path, arguments):
    app = serve_mod._build_http_app(_graph_file(tmp_path), json_response=True)
    with _client(app) as client:
        result = _call_tool_result(
            client,
            _init_session(client),
            "query_graph",
            arguments,
            rid=2,
        )
    assert result["isError"] is True
    assert result["structuredContent"] == {"code": "invalid_arguments"}


def _project_with_graph(tmp_path, node_count: int) -> str:
    """Create ``<proj>/graphify-out/graph.json`` and return the project dir."""
    proj = tmp_path / "proj"
    (proj / "graphify-out").mkdir(parents=True)
    graph = {
        "directed": True,
        "nodes": [{"id": f"n{i}", "label": f"N{i}", "community": 0} for i in range(node_count)],
        "edges": [],
    }
    (proj / "graphify-out" / "graph.json").write_text(json.dumps(graph), encoding="utf-8")
    return str(proj)


def _init_session(client) -> dict:
    init = client.post("/mcp", headers=_MCP_HEADERS, json=_INIT_BODY)
    assert init.status_code == 200
    headers = {**_MCP_HEADERS, "mcp-session-id": init.headers.get("mcp-session-id")}
    client.post(
        "/mcp", headers=headers, json={"jsonrpc": "2.0", "method": "notifications/initialized"}
    )
    return headers


def _call_tool(client, headers, name, arguments, rid) -> str:
    return _call_tool_result(client, headers, name, arguments, rid)["content"][0]["text"]


def _call_tool_result(client, headers, name, arguments, rid) -> dict:
    resp = client.post(
        "/mcp",
        headers=headers,
        json={
            "jsonrpc": "2.0",
            "id": rid,
            "method": "tools/call",
            "params": {"name": name, "arguments": arguments},
        },
    )
    assert resp.status_code == 200
    return resp.json()["result"]


def test_project_path_is_optional_on_every_tool(tmp_path):
    """Multi-project support is additive: every tool gains an optional
    project_path, and none of them makes it required."""
    app = serve_mod._build_http_app(_graph_file(tmp_path), json_response=True)
    with _client(app) as client:
        headers = _init_session(client)
        resp = client.post(
            "/mcp",
            headers=headers,
            json={"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
        )
        for tool in resp.json()["result"]["tools"]:
            props = tool["inputSchema"].get("properties", {})
            assert "project_path" in props, f"{tool['name']} missing project_path"
            assert "project_path" not in tool["inputSchema"].get("required", [])


def test_project_path_routes_to_that_projects_graph(tmp_path):
    """One running server answers against the default graph when project_path is
    omitted, and against a project's own graph when it is supplied."""
    proj = _project_with_graph(tmp_path, node_count=3)  # default graph has 2 nodes
    app = serve_mod._build_http_app(_graph_file(tmp_path), json_response=True)
    with _client(app) as client:
        headers = _init_session(client)
        assert "Nodes: 2" in _call_tool(client, headers, "graph_stats", {}, rid=2)
        assert "Nodes: 3" in _call_tool(
            client, headers, "graph_stats", {"project_path": proj}, rid=3
        )
        # Falling back to the default afterwards still works (no state leak).
        assert "Nodes: 2" in _call_tool(client, headers, "graph_stats", {}, rid=4)


def test_bad_project_path_errors_without_killing_server(tmp_path):
    """A missing project graph is a tool error, not a process exit — the server
    keeps serving the default graph."""
    app = serve_mod._build_http_app(_graph_file(tmp_path), json_response=True)
    with _client(app) as client:
        headers = _init_session(client)
        bad = _call_tool_result(
            client,
            headers,
            "graph_stats",
            {"project_path": str(tmp_path / "does-not-exist")},
            rid=2,
        )
        assert bad["isError"] is True
        assert bad["structuredContent"] == {"code": "graph_unavailable"}
        assert str(tmp_path) not in json.dumps(bad)
        assert "Nodes: 2" in _call_tool(client, headers, "graph_stats", {}, rid=3)


def test_project_path_outside_allowed_roots_is_refused_without_disclosure(tmp_path):
    allowed = tmp_path / "allowed"
    outside = tmp_path / "outside"
    allowed.mkdir()
    outside_project = _project_with_graph(outside, node_count=3)
    app = serve_mod._build_http_app(
        _graph_file(allowed), allowed_project_roots=[allowed], json_response=True
    )
    with _client(app) as client:
        result = _call_tool_result(
            client,
            _init_session(client),
            "graph_stats",
            {"project_path": outside_project},
            rid=2,
        )
    assert result["isError"] is True
    assert result["structuredContent"] == {"code": "project_access_denied"}
    assert str(outside) not in json.dumps(result)


def test_project_path_symlink_escape_is_refused(tmp_path, path_alias):
    allowed = tmp_path / "allowed"
    outside = tmp_path / "outside"
    allowed.mkdir()
    outside_project = Path(_project_with_graph(outside, node_count=3))
    escaped = path_alias(allowed / "linked-project", outside_project)
    app = serve_mod._build_http_app(
        _graph_file(allowed), allowed_project_roots=[allowed], json_response=True
    )
    with _client(app) as client:
        result = _call_tool_result(
            client,
            _init_session(client),
            "graph_stats",
            {"project_path": str(escaped)},
            rid=2,
        )
    assert result["isError"] is True
    assert result["structuredContent"] == {"code": "project_access_denied"}


def test_github_tools_are_absent_by_default(tmp_path):
    app = serve_mod._build_http_app(_graph_file(tmp_path), json_response=True)
    with _client(app) as client:
        headers = _init_session(client)
        response = client.post(
            "/mcp",
            headers=headers,
            json={"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
        )
    names = {tool["name"] for tool in response.json()["result"]["tools"]}
    assert {"list_prs", "get_pr_impact", "triage_prs"}.isdisjoint(names)


def test_github_tools_require_a_fixed_repository(tmp_path):
    with pytest.raises(ValueError, match="fixed GitHub repository"):
        serve_mod._build_http_app(
            _graph_file(tmp_path), enable_github_tools=True, json_response=True
        )


def test_github_tools_ignore_client_repository_and_use_fixed_authority(tmp_path, monkeypatch):
    captured = {}

    def fetch_prs(*, repo, base):
        captured.update(repo=repo, base=base)
        return []

    monkeypatch.setattr("graphify.prs.fetch_prs", fetch_prs)
    monkeypatch.setattr("graphify.prs.fetch_worktrees", lambda: {})
    app = serve_mod._build_http_app(
        _graph_file(tmp_path),
        enable_github_tools=True,
        github_repo="owner/fixed",
        json_response=True,
    )
    with _client(app) as client:
        result = _call_tool_result(
            client,
            _init_session(client),
            "list_prs",
            {"repo": "attacker/other", "base": "main"},
            rid=2,
        )
    assert result.get("isError", False) is False
    assert captured == {"repo": "owner/fixed", "base": "main"}


def test_http_triage_reports_worktree_presence_without_absolute_path(tmp_path, monkeypatch):
    leaked = str(tmp_path / "private-worktree")
    candidate = SimpleNamespace(
        number=1,
        branch="feature",
        base_branch="main",
        status="READY",
        ci_status="SUCCESS",
        review_decision="",
        days_old=1,
        author="alice",
        title="Change",
        blast_radius=0,
        files_changed=[],
        communities_touched=[],
        nodes_affected=0,
        worktree_path=None,
    )
    monkeypatch.setattr("graphify.prs.fetch_prs", lambda **_kwargs: [candidate])
    monkeypatch.setattr("graphify.prs.fetch_worktrees", lambda: {"feature": leaked})
    monkeypatch.setattr("graphify.prs.fetch_pr_files", lambda *_args, **_kwargs: [])
    app = serve_mod._build_http_app(
        _graph_file(tmp_path),
        enable_github_tools=True,
        github_repo="owner/fixed",
        json_response=True,
    )
    with _client(app) as client:
        result = _call_tool_result(
            client,
            _init_session(client),
            "triage_prs",
            {"base": "main"},
            rid=2,
        )
    rendered = json.dumps(result)
    assert "worktree=active" in rendered
    assert leaked not in rendered


def test_github_failure_is_protocol_error_without_raw_exception(tmp_path, monkeypatch):
    leaked = str(tmp_path / "credential.txt")

    def fail(**_kwargs):
        raise RuntimeError(f"credential failure at {leaked}")

    monkeypatch.setattr("graphify.prs.fetch_prs", fail)
    app = serve_mod._build_http_app(
        _graph_file(tmp_path),
        enable_github_tools=True,
        github_repo="owner/fixed",
        json_response=True,
    )
    with _client(app) as client:
        result = _call_tool_result(
            client,
            _init_session(client),
            "list_prs",
            {"base": "main"},
            rid=2,
        )
    assert result["isError"] is True
    assert result["structuredContent"] == {"code": "github_unavailable"}
    assert leaked not in json.dumps(result)


def test_unknown_tool_error_does_not_echo_client_input(tmp_path):
    supplied_name = str(tmp_path / "secret-tool")
    app = serve_mod._build_http_app(_graph_file(tmp_path), json_response=True)
    with _client(app) as client:
        result = _call_tool_result(
            client,
            _init_session(client),
            supplied_name,
            {},
            rid=2,
        )
    assert result["isError"] is True
    assert result["structuredContent"] == {"code": "unknown_tool"}
    assert supplied_name not in json.dumps(result)


def test_get_node_prefers_exact_match_over_earlier_substring_over_http(tmp_path):
    graph_path = tmp_path / "graph.json"
    graph_path.write_text(
        json.dumps(
            {
                "directed": True,
                "nodes": [
                    {"id": "foobar", "label": "Foobar helper", "community": 0},
                    {"id": "foo", "label": "Foo", "community": 0},
                ],
                "edges": [],
            }
        ),
        encoding="utf-8",
    )
    app = serve_mod._build_http_app(str(graph_path), json_response=True)
    with _client(app) as client:
        result = _call_tool_result(
            client,
            _init_session(client),
            "get_node",
            {"label": "foo"},
            rid=2,
        )

    assert result.get("isError", False) is False
    assert result["content"][0]["text"].startswith("Node: Foo\n  ID: foo\n")


def test_internal_tool_exception_is_client_safe_protocol_error(tmp_path, monkeypatch):
    leaked = str(tmp_path / "operator-secret.txt")

    def fail(*_args, **_kwargs):
        raise RuntimeError(f"failed beside {leaked}")

    monkeypatch.setattr("graphify.analyze.god_nodes", fail)
    app = serve_mod._build_http_app(_graph_file(tmp_path), json_response=True)
    with _client(app) as client:
        result = _call_tool_result(
            client,
            _init_session(client),
            "god_nodes",
            {},
            rid=2,
        )

    assert result["isError"] is True
    assert result["structuredContent"] == {"code": "internal_error"}
    assert leaked not in json.dumps(result)


def test_stdio_adapter_uses_lexically_bound_mcp_types_for_unknown_tool_error(tmp_path):
    graph_path = _graph_file(tmp_path)

    async def exercise() -> None:
        parameters = StdioServerParameters(
            command=sys.executable,
            args=["-m", "graphify.serve", graph_path],
            cwd=str(Path(__file__).resolve().parents[1]),
        )
        with (tmp_path / "stdio-stderr.log").open("w", encoding="utf-8") as stderr:
            async with stdio_client(parameters, errlog=stderr) as streams:
                async with ClientSession(*streams) as session:
                    await session.initialize()
                    result = await session.call_tool("not-a-tool", {})
        assert result.isError is True
        assert result.structuredContent == {"code": "unknown_tool"}
        content = result.content[0]
        assert isinstance(content, TextContent)
        assert content.text == "Unknown tool."

    asyncio.run(exercise())


def test_context_cache_is_bounded_by_entries_and_serialized_bytes():
    cache = serve_mod._BoundedContextCache(max_entries=2, max_bytes=10)
    cache.put("/a", key=(1, 4), graph=object(), communities={}, serialized_bytes=4)
    cache.put("/b", key=(1, 4), graph=object(), communities={}, serialized_bytes=4)
    cache.put("/c", key=(1, 4), graph=object(), communities={}, serialized_bytes=4)
    assert cache.keys() == ("/b", "/c")
    cache.put("/large", key=(1, 20), graph=object(), communities={}, serialized_bytes=20)
    assert cache.keys() == ("/b", "/c")

    graph = serve_mod.nx.Graph()
    graph.add_nodes_from(range(10))
    cache.put("/expanded", key=(1, 1), graph=graph, communities={}, serialized_bytes=1)
    assert "/expanded" not in cache.keys()


@pytest.mark.parametrize(
    "host",
    ["0.0.0.0", "::", "10.0.0.8", "93.184.216.34"],
)
def test_nonloopback_http_refuses_missing_api_key_before_app_start(tmp_path, host):
    with pytest.raises(ValueError, match="API key"):
        serve_mod._build_http_app(
            _graph_file(tmp_path), host=host, allowed_hosts=["graph.example"], json_response=True
        )


def test_wildcard_http_requires_explicit_allowed_host(tmp_path):
    with pytest.raises(ValueError, match="allowed Host"):
        serve_mod._build_http_app(
            _graph_file(tmp_path), host="0.0.0.0", api_key="secret", json_response=True
        )


def test_nonloopback_http_with_auth_and_host_validation_builds(tmp_path):
    app = serve_mod._build_http_app(
        _graph_file(tmp_path),
        host="0.0.0.0",
        api_key="secret",
        allowed_hosts=["graph.example"],
        json_response=True,
    )
    assert app is not None


def test_wildcard_http_enforces_allowed_host_header(tmp_path):
    app = serve_mod._build_http_app(
        _graph_file(tmp_path),
        host="0.0.0.0",
        api_key="secret",
        allowed_hosts=["graph.example"],
        json_response=True,
    )
    with TestClient(app, base_url="http://graph.example") as client:
        authorized = client.post(
            "/mcp",
            headers={**_MCP_HEADERS, "X-API-Key": "secret"},
            json=_INIT_BODY,
        )
        assert authorized.status_code == 200
        refused = client.post(
            "/mcp",
            headers={
                **_MCP_HEADERS,
                "Host": "attacker.example",
                "X-API-Key": "secret",
            },
            json=_INIT_BODY,
        )
        assert refused.status_code == 421


def test_serve_http_resolves_hostname_once_and_binds_validated_address(tmp_path, monkeypatch):
    calls = []

    def resolve(*_args, **_kwargs):
        calls.append(1)
        return [(2, 1, 6, "", ("127.0.0.1", 8080))]

    captured = {}
    monkeypatch.setattr("graphify.security.socket.getaddrinfo", resolve)
    monkeypatch.setattr("uvicorn.run", lambda app, **kwargs: captured.update(kwargs))

    serve_mod.serve_http(_graph_file(tmp_path), host="bind.example")

    assert len(calls) == 1
    assert captured["host"] == "127.0.0.1"


def test_nonloopback_serve_requires_tls_before_uvicorn(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "uvicorn.run",
        lambda *_args, **_kwargs: pytest.fail("insecure listener must not start"),
    )
    with pytest.raises(ValueError, match="TLS certificate and key"):
        serve_mod.serve_http(
            _graph_file(tmp_path),
            host="10.0.0.8",
            api_key="secret",
            allowed_hosts=["graph.example"],
        )


def test_nonloopback_serve_passes_tls_material_to_uvicorn(tmp_path, monkeypatch):
    captured = {}
    cert = tmp_path / "cert.pem"
    key = tmp_path / "key.pem"
    cert.write_text("certificate", encoding="utf-8")
    key.write_text("key", encoding="utf-8")
    monkeypatch.setattr("uvicorn.run", lambda app, **kwargs: captured.update(kwargs))

    serve_mod.serve_http(
        _graph_file(tmp_path),
        host="10.0.0.8",
        api_key="secret",
        allowed_hosts=["graph.example"],
        ssl_certfile=str(cert),
        ssl_keyfile=str(key),
    )

    assert captured["ssl_certfile"] == str(cert)
    assert captured["ssl_keyfile"] == str(key)


def test_stateless_mode_initialize(tmp_path):
    app = serve_mod._build_http_app(_graph_file(tmp_path), stateless=True, json_response=True)
    with _client(app) as client:
        resp = client.post("/mcp", headers=_MCP_HEADERS, json=_INIT_BODY)
        assert resp.status_code == 200


def test_stateless_with_timeout_does_not_raise(tmp_path):
    # session_timeout must be forced to None in stateless mode (the SDK raises
    # RuntimeError otherwise). Building + a request should just work.
    app = serve_mod._build_http_app(
        _graph_file(tmp_path), stateless=True, session_timeout=3600, json_response=True
    )
    with _client(app) as client:
        assert client.post("/mcp", headers=_MCP_HEADERS, json=_INIT_BODY).status_code == 200


def test_session_timeout_zero_disables(tmp_path):
    # 0 / non-positive must disable reaping without tripping the SDK's validation.
    app = serve_mod._build_http_app(_graph_file(tmp_path), session_timeout=0, json_response=True)
    with _client(app) as client:
        assert client.post("/mcp", headers=_MCP_HEADERS, json=_INIT_BODY).status_code == 200


# --- CLI argument parsing -------------------------------------------------


def test_cli_defaults_to_stdio(monkeypatch):
    calls = {}
    monkeypatch.setattr(
        serve_mod, "serve", lambda gp, **kwargs: calls.setdefault("stdio", (gp, kwargs))
    )
    monkeypatch.setattr(serve_mod, "serve_http", lambda *a, **k: calls.setdefault("http", (a, k)))
    serve_mod._main(["graphify-out/graph.json"])
    graph_path, options = calls["stdio"]
    assert graph_path == "graphify-out/graph.json"
    assert options == {
        "allowed_project_roots": None,
        "enable_github_tools": False,
        "github_repo": None,
    }
    assert "http" not in calls


def test_cli_http_passes_flags(monkeypatch):
    captured = {}
    monkeypatch.setattr(serve_mod, "serve", lambda gp: captured.setdefault("stdio", gp))
    monkeypatch.setattr(serve_mod, "serve_http", lambda gp, **k: captured.update(gp=gp, **k))
    serve_mod._main(
        [
            "g.json",
            "--transport",
            "http",
            "--host",
            "0.0.0.0",
            "--port",
            "9000",
            "--api-key",
            "k",
            "--ssl-certfile",
            "cert.pem",
            "--ssl-keyfile",
            "key.pem",
            "--allow-host",
            "graph.example",
            "--allow-project-root",
            "/srv/projects",
            "--enable-github-tools",
            "--github-repo",
            "owner/repo",
            "--stateless",
        ]
    )
    assert captured["gp"] == "g.json"
    assert captured["host"] == "0.0.0.0"
    assert captured["port"] == 9000
    assert captured["api_key"] == "k"
    assert captured["ssl_certfile"] == "cert.pem"
    assert captured["ssl_keyfile"] == "key.pem"
    assert captured["allowed_hosts"] == ["graph.example"]
    assert captured["allowed_project_roots"] == ["/srv/projects"]
    assert captured["enable_github_tools"] is True
    assert captured["github_repo"] == "owner/repo"
    assert captured["stateless"] is True


def test_cli_api_key_from_env(monkeypatch):
    captured = {}
    monkeypatch.setenv("GRAPHIFY_API_KEY", "from-env")
    monkeypatch.setattr(serve_mod, "serve_http", lambda gp, **k: captured.update(**k))
    serve_mod._main(["g.json", "--transport", "http"])
    assert captured["api_key"] == "from-env"
