"""Compatibility wrappers for authoritative serialized Graphify state."""

from __future__ import annotations

import sys
from collections.abc import Mapping
from pathlib import Path

import networkx as nx

from .graph_state import GRAPHIFY_PROFILE_KEY as GRAPHIFY_PROFILE_KEY
from .graph_state import DecodeMode, GraphState, decode_graph_state, decode_graph_state_file
from .multigraph_compat import require_multigraph_capabilities


def load_graph(
    data: object,
    *,
    require_capabilities: bool = True,
) -> nx.Graph | nx.DiGraph | nx.MultiDiGraph:
    """Load current or losslessly repairable legacy state for read-only use.

    Stateful paths must use ``load_graph_state_file`` with an explicit decode
    mode and require complete class declarations before writing. This reader
    accepts legacy omissions only when the state codec can repair them without
    discarding a record; malformed nodes, edges, and metadata are refused.
    """
    if not isinstance(data, Mapping):
        raise TypeError("serialized graph data must be a JSON object")
    state = decode_graph_state(data, mode=DecodeMode.READ_ONLY_LEGACY)
    if state.graph_type == "multidigraph" and require_capabilities:
        require_multigraph_capabilities()
    for diagnostic in state.diagnostics:
        print(f"[graphify] WARNING: {diagnostic.message}", file=sys.stderr)
    return state.graph


def load_graph_file(
    path: str | Path,
    *,
    require_capabilities: bool = True,
) -> nx.Graph | nx.DiGraph | nx.MultiDiGraph:
    """Load a size-capped graph file through the authoritative state codec."""
    state = load_graph_state_file(path, mode=DecodeMode.READ_ONLY_LEGACY)
    if state.graph_type == "multidigraph" and require_capabilities:
        require_multigraph_capabilities()
    _surface_diagnostics(state)
    return state.graph


def _surface_diagnostics(state: GraphState) -> None:
    for diagnostic in state.diagnostics:
        print(f"[graphify] WARNING: {diagnostic.message}", file=sys.stderr)


def load_graph_state_file(
    path: str | Path,
    *,
    mode: DecodeMode,
    enforce_size_cap: bool = True,
) -> GraphState:
    """Load full state while preserving metadata, hyperedges, and diagnostics."""
    state = decode_graph_state_file(path, mode=mode, enforce_size_cap=enforce_size_cap)
    if state.graph_type == "multidigraph":
        require_multigraph_capabilities()
    _surface_diagnostics(state)
    return state
