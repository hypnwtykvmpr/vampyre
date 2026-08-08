"""C# receiver-typed member-call resolution (#1609).

`recv.Method()` where `recv` is a typed field / property / parameter / local must
resolve to the receiver TYPE's method — not a bare same-named match. Before this,
C# had no member-call resolver: the bare method name matched any same-named method
in the corpus, so `_server.Save()` silently mis-bound to an unrelated `Cache.Save()`
(a WRONG edge, not just a missing one). Resolution is by receiver type with the
single-definition god-node guard; an untypable receiver produces no edge.
"""

from __future__ import annotations

import os
from pathlib import Path

from graphify.extract import extract


def _calls(tmp_path, files: dict[str, str]):
    for name, body in files.items():
        p = tmp_path / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(body, encoding="utf-8")
    old = os.getcwd()
    try:
        os.chdir(tmp_path)
        r = extract([Path(n) for n in files], cache_root=tmp_path / ".cache")
    finally:
        os.chdir(old)
    calls = {(e["source"], e["target"]) for e in r["edges"] if e["relation"] == "calls"}
    return calls, r


_AMBIG = {
    "S.cs": (
        "public class Server { public bool Save() => true; }\n"
        "public class Cache  { public bool Save() => false; }\n"
        "public class Repo {\n"
        "    private Server _server = new Server();\n"
        "    public bool Commit() { return _server.Save(); }\n"
        "}\n"
    )
}


def _find(r, label, id_contains):
    return next(n["id"] for n in r["nodes"] if n["label"] == label and id_contains in n["id"])


def _method_id(result: dict, owner_label: str, method_label: str) -> str:
    owners = {node["id"] for node in result["nodes"] if node.get("label") == owner_label}
    labels = {node["id"]: node.get("label") for node in result["nodes"]}
    matches = {
        edge["target"]
        for edge in result["edges"]
        if edge.get("relation") == "method"
        and edge.get("source") in owners
        and labels.get(edge.get("target")) == method_label
    }
    assert len(matches) == 1, (owner_label, method_label, matches)
    return next(iter(matches))


def test_field_receiver_resolves_to_declared_type_not_bare_match(tmp_path):
    calls, r = _calls(tmp_path, _AMBIG)
    commit = _find(r, ".Commit()", "commit")
    server_save = _find(r, ".Save()", "server")
    cache_save = _find(r, ".Save()", "cache")
    assert (commit, server_save) in calls, "field.Method() must resolve to the field's type"
    assert (commit, cache_save) not in calls, "must NOT mis-bind to an unrelated same-named method"


def test_parameter_receiver_resolves(tmp_path):
    calls, r = _calls(
        tmp_path,
        {
            "S.cs": (
                "public class Server { public bool Save() => true; }\n"
                "public class Cache  { public bool Save() => false; }\n"
                "public class Svc { public static bool Copy(Server server) { return server.Save(); } }\n"
            )
        },
    )
    copy = _method_id(r, "Svc", ".Copy()")
    server_save = _method_id(r, "Server", ".Save()")
    cache_save = _method_id(r, "Cache", ".Save()")
    assert (copy, server_save) in calls
    assert (copy, cache_save) not in calls


def test_local_var_receiver_resolves(tmp_path):
    calls, r = _calls(
        tmp_path,
        {
            "S.cs": (
                "public class Server { public bool Save() => true; }\n"
                "public class R {\n"
                "    public bool A() { Server s = new Server(); return s.Save(); }\n"
                "    public bool B() { var v = new Server(); return v.Save(); }\n"
                "}\n"
            )
        },
    )
    server_save = _method_id(r, "Server", ".Save()")
    assert (_method_id(r, "R", ".A()"), server_save) in calls, "explicit-typed local"
    assert (_method_id(r, "R", ".B()"), server_save) in calls, "var = new T() local"


def test_cross_file_receiver_resolves(tmp_path):
    calls, r = _calls(
        tmp_path,
        {
            "Server.cs": (
                "public class Server { public bool Save() => true; }\n"
                "public class Cache  { public bool Save() => false; }\n"
            ),
            "Repo.cs": (
                "public class Repo { private Server _s = new Server(); "
                "public bool Commit() { return _s.Save(); } }\n"
            ),
        },
    )
    commit = _method_id(r, "Repo", ".Commit()")
    assert (commit, _method_id(r, "Server", ".Save()")) in calls
    assert (commit, _method_id(r, "Cache", ".Save()")) not in calls


def test_this_and_static_receivers(tmp_path):
    calls, r = _calls(
        tmp_path,
        {
            "S.cs": (
                "public class Util { public static int F() => 1; }\n"
                "public class R {\n"
                "    public bool A() { return this.B(); }\n"
                "    public bool B() => true;\n"
                "    public int G() { return Util.F(); }\n"
                "}\n"
            )
        },
    )
    assert (_method_id(r, "R", ".A()"), _method_id(r, "R", ".B()")) in calls, "this.B() -> R.B"
    assert (_method_id(r, "R", ".G()"), _method_id(r, "Util", ".F()")) in calls, (
        "Util.F() -> Util.F"
    )


def test_untyped_receiver_emits_no_edge(tmp_path):
    calls, r = _calls(
        tmp_path,
        {
            "S.cs": (
                "public class Server { public bool Save() => true; }\n"
                "public class R { public bool C(dynamic x) { return x.Save(); } }\n"
            )
        },
    )
    caller = _method_id(r, "R", ".C()")
    server_save = _method_id(r, "Server", ".Save()")
    assert (caller, server_save) not in calls, "dynamic receiver must not resolve"


def test_method_absent_on_type_emits_no_edge(tmp_path):
    calls, r = _calls(
        tmp_path,
        {
            "S.cs": (
                "public class Server { public bool Save() => true; }\n"
                "public class R { private Server _s = new Server(); "
                "public bool C() { return _s.Missing(); } }\n"
            )
        },
    )
    caller = _method_id(r, "R", ".C()")
    server_save = _method_id(r, "Server", ".Save()")
    assert (caller, server_save) not in calls


def test_unqualified_call_still_resolves(tmp_path):
    calls, r = _calls(
        tmp_path,
        {
            "S.cs": (
                "public class R { public bool A() { Helper(); return true; } "
                "private void Helper() {} }\n"
            )
        },
    )
    assert (_method_id(r, "R", ".A()"), _method_id(r, "R", ".Helper()")) in calls
