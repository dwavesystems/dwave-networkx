# Copyright 2026 D-Wave
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import itertools
from collections import namedtuple
from typing import Callable, Literal, get_args

import networkx as nx
import numpy as np
from dwave.embedding import is_valid_embedding, verify_embedding

from dwave_networkx import zephyr_coordinates, zephyr_graph

ZephyrNode = tuple[int, int, int, int, int]  # (u, w, k, j, z) coordinate tuple
Embedding = dict[ZephyrNode, ZephyrNode]  # Internal single-node format
EmbeddingChain = dict[ZephyrNode, tuple[ZephyrNode, ...]]  # External chain format
YieldType = Literal["node", "edge", "rail-edge"]
QuotientSearchType = Literal["by_quotient_rail", "by_quotient_node", "by_rail_then_node"]

ZephyrSearchMetadata = namedtuple(
    "ZephyrSearchMetadata", ["max_num_yielded", "starting_num_yielded", "final_num_yielded"]
)


def _validate_graph_inputs(source: nx.Graph, target: nx.Graph):
    """Validate that source and target are Zephyr NetworkX graphs.

    Both source and target graphs must be networkx graph instances with a 'family' metadata key
    set to 'zephyr'. Each graph must also contain 'rows', 'tile' and 'labels metadata keys.

    Args:
        source (nx.Graph): Source Zephyr graph.
        target (nx.Graph): Target Zephyr graph.

    Raises:
        TypeError: If inputs are not NetworkX graphs.
        ValueError: If either graph is not a Zephyr family graph or is missing 'rows'/'tile'
            metadata.
    """
    if not isinstance(source, nx.Graph) or not isinstance(target, nx.Graph):
        raise TypeError("source and target must both be networkx.Graph instances")

    if source.graph.get("family") != "zephyr":
        raise ValueError("source graph should be a zephyr family graph")
    if target.graph.get("family") != "zephyr":
        raise ValueError("target graph should be a zephyr family graph")

    for graph_name, graph in zip(("source", "target"), (source, target)):
        for key in ("rows", "tile", "labels"):
            if key not in graph.graph:
                raise ValueError(f"{graph_name} graph is missing required '{key}' metadata")


def _extract_graph_properties(source: nx.Graph, target: nx.Graph) -> tuple[int, int, int]:
    """Extract and validate Zephyr graph properties, returning ``(m, tp, t)``.

    Each graph must contain required metadata fields: 'rows' (number of rows) and 'tile'
    (tile count). All metadata values must be positive integers. The source and target graphs must
    have matching row counts. The target tile count must be greater than or equal to the source tile
    count to accommodate the embedding.

    Args:
        source (nx.Graph): Source Zephyr graph.
        target (nx.Graph): Target Zephyr graph.

    Returns:
        tuple[int, int, int]: ``(m, tp, t)`` where ``m`` is rows,
            ``tp`` is source tile count, and ``t`` is target tile count.

    Raises:
        TypeError: If metadata values are not integers.
        ValueError: If graph metadata is missing or incompatible.
    """
    m = source.graph["rows"]
    tp = source.graph["tile"]
    t = target.graph["tile"]

    for v, name in zip((m, tp, t), ("rows", "source tile", "target tile")):
        if not isinstance(v, int):
            raise TypeError(f"graph '{name}' metadata must be an integer")
        if v <= 0:
            raise ValueError(f"graph '{name}' metadata must be positive")
    if target.graph["rows"] != m:
        raise ValueError("source and target must have the same number of rows")
    if t < tp:
        raise ValueError("target tile count must be >= source tile count")

    return m, tp, t


def _validate_search_parameters(
    quotient_search: str,
    yield_type: str,
    embedding: EmbeddingChain | None,
) -> None:
    """Validate high-level search parameters.

    ``quotient_search`` must be one of ``'by_quotient_rail'``, ``'by_quotient_node'``, or
    ``'by_rail_then_node'``; ``yield_type`` must be one of ``'node'``, ``'edge'``, or
    ``'rail-edge'``; and ``embedding`` must be ``None`` or a ``dict`` representing a
    one-to-one chain mapping of Zephyr coordinate nodes, where each value is a tuple
    of one target nodes.

    Args:
        quotient_search (str): Search mode.
        yield_type (str): Optimization objective.
        embedding (EmbeddingChain | None): Optional initial one-to-one chain mapping.

    Raises:
        ValueError: If ``quotient_search`` or ``yield_type`` is invalid, or if ``embedding``
            contains duplicate target nodes (i.e. is not one-to-one).
        TypeError: If ``embedding`` is invalid.
    """
    valid_ksearch = get_args(QuotientSearchType)
    valid_yield_type = get_args(YieldType)

    if quotient_search not in valid_ksearch:
        raise ValueError(f"quotient_search must be one of {sorted(valid_ksearch)}. Got "
                         f"'{quotient_search}'")
    if yield_type not in valid_yield_type:
        raise ValueError(
            f"yield_type must be one of {sorted(valid_yield_type)}. Got '{yield_type}'"
        )
    if embedding is not None and not isinstance(embedding, dict):
        raise TypeError(f"embedding must be a dictionary when provided. Got {type(embedding)}")
    if embedding is not None:
        # Validate chain format: keys are nodes, values are tuples of nodes
        for key, value in embedding.items():
            if not isinstance(key, tuple):
                raise TypeError(
                    f"embedding keys must be tuples representing nodes. Got key {key} "
                    f"of type {type(key)}"
                )
            if len(key) != 5:
                raise ValueError(
                    f"embedding keys must be 5-tuples representing Zephyr coordinates. "
                    f"Got key {key} of length {len(key)}"
                )
            if not isinstance(value, tuple):
                raise TypeError(
                    f"embedding values must be tuples representing node chains. Got value {value} "
                    f"of type {type(value)}"
                )
            if len(value) == 0:
                raise ValueError(
                    f"embedding chains must be non-empty. Got empty tuple for key {key}"
                )
            elif len(value) != 1:
                raise ValueError(
                    f"embedding chains must contain exactly one target node for each source node. "
                    f"Got chain of length {len(value)} for key {key}. Chain is: {value}"
                )
            for i, node in enumerate(value):
                if not isinstance(node, tuple) or len(node) != 5:
                    raise ValueError(
                        f"embedding chains must contain 5-tuples. Got {node} "
                        f"(length {len(node) if isinstance(node, tuple) else 'N/A'}) "
                        f"at position {i} in chain for key {key}"
                    )
        # Check one-to-one constraint: flatten all chains and ensure no duplicates
        all_target_nodes = []
        for chain in embedding.values():
            all_target_nodes.extend(chain)
        if len(all_target_nodes) != len(set(all_target_nodes)):
            raise ValueError(
                "embedding must be a one-to-one mapping: duplicate target nodes detected across "
                "chains. "
            )


def _ensure_coordinate_source(
    source: nx.Graph,
    m: int,
    tp: int,
) -> tuple[nx.Graph, set[ZephyrNode], Callable[[ZephyrNode], int | ZephyrNode]]:
    """Normalise the source graph to coordinate labels.

    This function ensures the rest of the search code can operate on a
    coordinate-labelled representation of the graphs, regardless of the input node-labelling
    convention. The quotient search internally assumes Zephyr coordinates of the
    form ``(u, w, k, j, z)``.

    Args:
        source (nx.Graph): Source Zephyr graph, either linear or coordinate labelled.
        m (int): Number of rows (must be consistent with ``source``).
        tp (int): Source tile count (must be consistent with ``source``).

    Returns:
        tuple[nx.Graph, set[ZephyrNode], Callable[[ZephyrNode], int | ZephyrNode]]:
            ``(_source, source_nodes, to_source)`` where ``_source`` is
            coordinate-labelled, ``source_nodes`` is the full canonical coordinate
            node set implied by ``m`` and ``tp``, and ``to_source`` maps
            coordinate nodes back to the original source labelling space.

    Raises:
        ValueError: If source labels are unsupported.
    """
    source_nodes: set[ZephyrNode] = {
        (u, w, k, j, z)
        for u in range(2)
        for w in range(2 * m + 1)
        for k in range(tp)
        for j in range(2)
        for z in range(m)
    }

    # If the labels are linear integers, convert to coordinate labels and define a function to
    # convert back.
    if source.graph["labels"] == "int":
        coords = zephyr_coordinates(m, tp)
        to_tuple = coords.linear_to_zephyr
        _source = zephyr_graph(
            m,
            tp,
            coordinates=True,
            node_list=source_nodes,
            edge_list=[(to_tuple(n1), to_tuple(n2)) for n1, n2 in source.edges()],
        )

        def to_source_linear(n: ZephyrNode) -> int:
            return coords.zephyr_to_linear(n)

        return _source, source_nodes, to_source_linear

    # IF labels are not linear nor coordinate, we raise an error.
    if source.graph["labels"] != "coordinate":
        raise ValueError("source graph has unknown labelling scheme")

    _source = source.copy()
    # Just in case the source graph is missing some nodes, we add the full canonical set implied by
    # m and tp. However, this should not happen. Discuss with Jack why would this ever be needed.
    # TODO: ask.
    _source.add_nodes_from(source_nodes)

    # If the labels are coordinate. Then we just return the graph as is and the identity function
    # for to_source:

    def to_source(n: ZephyrNode) -> ZephyrNode:
        return n

    return _source, source_nodes, to_source


def _ensure_coordinate_target(
    target: nx.Graph,
    m: int,
    t: int,
) -> tuple[nx.Graph, Callable[[ZephyrNode], int | ZephyrNode]]:
    """Return a coordinate-labelled target graph and conversion callable.

    This helper normalises ``target`` to coordinate labels and returns a callable that maps
    candidate nodes into the target's original label space.

    Similar to ``_ensure_coordinate_source``, but it does not return the full canonical node set
    because the search only checks node presence in the target rather than iterating over all
    nodes, and the target may be defective and missing some nodes.

    Args:
        target (nx.Graph): Target Zephyr graph, either linear or coordinate labelled.
        m (int): Number of rows (must be consistent with ``target``).
        t (int): Target tile count (must be consistent with ``target``).

    Returns:
        tuple[nx.Graph, Callable[[ZephyrNode], int | ZephyrNode]]:
            ``(_target, to_target)`` where ``_target`` is coordinate-labelled.

    Raises:
        ValueError: If target labels are unsupported.
    """
    if target.graph["labels"] == "int":
        coords = zephyr_coordinates(m, t)
        to_tuple = coords.linear_to_zephyr
        _target = zephyr_graph(
            m,
            t,
            coordinates=True,
            node_list=[to_tuple(n) for n in target.nodes()],
            edge_list=[(to_tuple(n1), to_tuple(n2)) for n1, n2 in target.edges()],
        )

        def to_target_linear(n: ZephyrNode) -> int:
            return coords.zephyr_to_linear(n)

        return _target, to_target_linear

    if target.graph["labels"] != "coordinate":
        raise ValueError("target graph has unknown labelling scheme")

    def to_target(n: ZephyrNode) -> ZephyrNode:
        return n

    return target, to_target


def _boundary_proposals(
    u: int,
    w: int,
    tp: int,
    t: int,
    embedding: Embedding,
    j: int = 0,
    z: int = 0,
) -> set[ZephyrNode]:
    r"""Generate candidate targets for boundary expansion.

    For a fixed quotient index ``(u, w, j, z)``, this function proposes all target ``k`` locations
    in that rail, then removes the entries already occupied by the currently mapped source
    :math:`k \in \{0, \dots, tp-1\}`.

    Args:
        u (int): Zephyr orientation.
        w (int): Zephyr column index.
        tp (int): Source tile count.
        embedding (Embedding): Current one-to-one proposal mapping.
        j (int): Intra-cell orientation index. Default is 0.
        z (int): Row index. Default is 0.

    Returns:
        set[ZephyrNode]: Available target coordinates with fixed ``(u, w, j, z)``.
    """
    all_target_coordinates = {(u, w, k, j, z) for k in range(t)}
    used_coordinates = {
        embedding[(u, w, k, j, z)]
        for k in range(tp)
        if (u, w, k, j, z) in embedding
    }
    return all_target_coordinates.difference(used_coordinates)


def _node_search(
    source: nx.Graph,
    target: nx.Graph,
    embedding: Embedding,
    *,
    expand_boundary_search: bool = True,
    ksymmetric: bool = False,
    yield_type: YieldType = "edge",
) -> Embedding:
    r"""Greedy node-level quotient search over Zephyr coordinates.

    The source and target are viewed in quotient blocks indexed by :math:`(u, w, j, z)`, each
    containing :math:`tp` source nodes. For each block, we propose target nodes with the same
    :math:`(u, w, j, z)` and varying target :math:`k`, optionally augmented with boundary proposals.

    The scoring objective is:

    .. math::

        \operatorname{score}(p) =
        \begin{cases}
        \sum\limits_{n \in B} \mathbf{1}[p_n \in V(T)] & \text{node yield}\\
        \sum\limits_{(n,m) \in E(S_B, S_\text{fixed})}
        \mathbf{1}[(p_n, \phi(m)) \in E(T)] & \text{edge yield}
        \end{cases}

    For a fixed quotient index :math:`q = (u, w, j, z)`, define the source block :math:`B_q` as

    .. math::

        B_q = \{(u, w, k, j, z) : k \in \{0, \dots, tp-1\}\}.

    A proposal :math:`p` is an assignment on that block, :math:`p: B_q \to V(T)`, and can be
    viewed as a length-``tp`` vector :math:`(p_0, \dots, p_{tp-1})` where :math:`p_k` is the
    proposed target node for source node :math:`(u, w, k, j, z)`.

    Here :math:`T` is the target graph, :math:`V(T)` is its node set, and :math:`E(T)` is its edge
    set. Let :math:`S` be the source graph and define the already-fixed outside set

    .. math::

        F_q = \{m \in V(S) \setminus B_q : m \in \operatorname{dom}(\phi)\},

    where :math:`\phi` is the current embedding`. Then

    .. math::

        E(S_B, S_\text{fixed})
        := \{(n,m) \in E(S) : n \in B_q,\ m \in F_q\},

    i.e., the source edges that cross from the current block to already-fixed source nodes outside
    the block.

    In other words, node yield counts how many proposed nodes :math:`p_n` are present in
    :math:`V(T)`; while edge yield counts how many source edges crossing from the current block to
    already-fixed nodes are preserved as target edges :math:`(p_n, \phi(m)) \in E(T)`.

    Yield types in this node-level search are interpreted as follows: ``"node"`` maximises target
    node presence for each proposed block; ``"edge"`` maximises preserved cross-block
    source-to-fixed edge connectivity; and ``"rail-edge"`` follows the same node-level scoring as
    ``"edge"`` in this function (the distinction between ``"edge"`` and ``"rail-edge"`` is made
    in rail-level search).

    Args:
        source (nx.Graph): Coordinate-labeled source Zephyr graph.
        target (nx.Graph): Coordinate-labeled target Zephyr graph.
        embedding (Embedding): Current mapping, updated in-place.
        expand_boundary_search (bool): If ``True``, augment boundary columns using the adjacent
            internal column. Defaults to ``True``.
        ksymmetric (bool): If ``True``, assume the order of source ``k`` indices is interchangeable
            for scoring and use top-``tp`` selection. Defaults to ``False``.
        yield_type (YieldType): ``"node"``, ``"edge"``, or ``"rail-edge"``. Defaults to ``"edge"``.

    Returns:
        Embedding: Updated embedding.

    Raises:
        ValueError: If graph geometry metadata is inconsistent.
    """
    m = source.graph["rows"]
    tp = source.graph["tile"]
    t = target.graph["tile"]
    if m != target.graph["rows"]:
        raise ValueError("source and target rows must match for node search")

    if expand_boundary_search:
        # Visit interior columns first so boundary expansion can reuse already-assigned assignments:
        iterator = itertools.product(
            range(2),
            list(range(1, 2 * m)) + [0, 2 * m],
            range(2),
            range(m),
        )
        ksymmetric_original = ksymmetric
    else:
        iterator = itertools.product(range(2), range(2 * m + 1), range(2), range(m))

    for u, w, j, z in iterator:
        # Base proposals preserve (u, w, j, z) and search only over target k-indices:
        proposals = [(u, w, k, j, z) for k in range(t)]

        if expand_boundary_search:
            if w == 0:
                ksymmetric = False
                # brrow candidates from adjacent internal column
                proposals += list(_boundary_proposals(u, 1, tp, t, embedding, j, z))
            elif w == 2 * m:
                ksymmetric = False
                proposals += list(_boundary_proposals(u, 2 * m - 1, tp, t, embedding, j, z))
            else:
                ksymmetric = ksymmetric_original

        if ksymmetric or yield_type != "edge":
            if yield_type == "node":
                # symmetry doesn't matter: just count how many proposed nodes are present in the
                # target:
                counts = [int(target.has_node(n_t)) for n_t in proposals]
            else:
                # Count preserved edges from already-mapped neighboring source nodes into each
                # proposed target node.
                source_neighbours = source.neighbors((u, w, 0, j, z))
                counts = [
                    sum(
                        int(target.has_edge(embedding[n_s], n_t))
                        for n_s in source_neighbours
                        if n_s in embedding
                    )
                    for n_t in proposals
                ]
            # performance: this is faster than selected = proposals[np.argsort()]...
            top_indices = np.argpartition(np.asarray(counts), -tp)[-tp:]
            selected = [proposals[idx] for idx in top_indices]
        else:
            # Nodes with different k indices in the source block are not interchangeable, so we
            # evaluate all permutations of the proposals:
            permutation_scores = {
                proposal_perm: sum(
                    int(target.has_edge(embedding[n], proposal_perm[k]))
                    for k in range(tp)
                    for n in source.neighbors((u, w, k, j, z))
                    if n in embedding
                )
                for proposal_perm in itertools.permutations(proposals, tp)
            }
            selected_key = max(permutation_scores, key=lambda k: permutation_scores[k])
            selected = list(selected_key)

        embedding.update(
            {(u, w, k, j, z): proposal for k, proposal in zip(range(tp), selected)}
        )

    return embedding


def _rail_search(
    source: nx.Graph,
    target: nx.Graph,
    embedding: Embedding,
    *,
    expand_boundary_search: bool = True,
    ksymmetric: bool = False,
    yield_type: YieldType = "edge",
) -> Embedding:
    r"""Greedy rail-level quotient search over Zephyr rails.

    A Zephyr rail is indexed by :math:`(u, w, k)` and contains nodes
    :math:`(u, w, k, j, z)` for :math:`j \in \{0,1\}` and :math:`z \in \{0,\dots,m-1\}`.

    For fixed orientation and column :math:`(u, w)`, define the source rail family

    .. math::

        \mathcal{R}^{S}_{u,w} := \{(u, w, k_s) : k_s \in \{0, \dots, t_p-1\}\}.

    The search chooses :math:`t_p` target rails for each family :math:`\mathcal{R}^{S}_{u,w}`
    from candidate rails optionally augmented at boundaries (:math:`w=0` and :math:`w=2m`) using
    adjacent interior columns.

    Let the target rail indexed by :math:`(u, w_t, k_t)` be

    .. math::

        R^{T}_{u,w_t,k_t} :=
        \{(u, w_t, k_t, j, z) : j \in \{0,1\},\ z \in \{0,\dots,m-1\}\}.

    We can define its objective for ``yield_type='edge'`` as the number of edges preserved within
    that rail, i.e., the numbe of edges in the target subgraph induced by the proposed rail, or
    equivalently the number of edges in the source rail (which is fixed) that are preserved by the
    proposal:

    .. math::

        Q(u,w_t,k_t) := |E(T[R^{T}_{u,w_t,k_t}])|,

    or, for ``yield_type='node'``, the number of present target nodes in that rail. Here :math:`T`
    is the target graph and :math:`E(T[R])` is the edge set of the target subgraph induced by node
    set :math:`R`.
    For ``yield_type='edge'``, each proposal also gets an external connectivity term counting
    preserved edges from already-embedded neighbouring source nodes into the proposed target rail.

    .. math::

        \operatorname{score}(u,w_t,k_t)
        = Q(u,w_t,k_t)
        + \sum \mathbf{1}[\text{external source edge maps to a target edge}].

    Depending on ``ksymmetric``, the algorithm either selects the top :math:`t_p` rail proposals by
    score (treating source :math:`k` order as interchangeable), or evaluates permutations assigning
    proposal rails to source indices :math:`k_s \in \{0,\dots,t_p-1\}`.

    Yield types in this rail-level search are interpreted as follows: ``"node"`` scores each
    proposal rail by the number of present target nodes in that rail. ``"edge"`` prefers rails
    that both have many internal rail edges and connect well to already-embedded neighbouring
    rails. ``"rail-edge"`` focuses first on how good the rail itself is, measured by the number of
    target edges inside that rail; when permutations are evaluated, it also includes the same
    already-embedded neighbour consistency term as ``"edge"``.

    Example: suppose two candidate target rails have the same internal rail structure, but one of
    them has more edges to neighbouring rails that are already fixed in the embedding. Then
    ``"edge"`` prefers that better-connected rail, while ``"rail-edge"`` treats the two rails as
    equivalent in the top-rail selection path because it only compares their internal rail
    structure there.

    Selected rails are then expanded back to node assignments for all :math:`(j,z)` in
    each source rail.

    Args:
        source (nx.Graph): Coordinate-labeled source Zephyr graph.
        target (nx.Graph): Coordinate-labeled target Zephyr graph.
        embedding (Embedding): Current mapping, updated in-place.
        expand_boundary_search (bool): If ``True``, include adjacent-column rail proposals when
            :math:`w` is at a boundary. Defaults to ``True``.
        ksymmetric (bool): If ``True``, treat source :math:`k` order as interchangeable when scoring
            rails. Defaults to ``False``.
        yield_type (str): ``"node"``, ``"edge"``, or ``"rail-edge"``. Defaults to ``"edge"``.

    Returns:
        Embedding: Updated embedding.

    Raises:
        ValueError: If duplicate target assignments are produced.
    """
    m = source.graph["rows"]
    tp = source.graph["tile"]
    t = target.graph["tile"]

    if yield_type == "node":
        rail_score = {
            (u, w, k): sum(target.has_node((u, w, k, j, z)) for j in range(2) for z in range(m))
            for u in range(2)
            for w in range(2 * m + 1)
            for k in range(t)
        }
    else:
        # Precompute per-rail edge number for fast proposal scoring.
        rail_score = {
            (u, w, k): target.subgraph(
                {(u, w, k, j, z) for j in range(2) for z in range(m)}
            ).number_of_edges()
            for u in range(2)
            for w in range(2 * m + 1)
            for k in range(t)
        }

    # when optimising for edges, we consider all edges that do not share the same orientation
    source_external_edges = source.edge_subgraph(
        {e for e in source.edges() if e[0][0] != e[1][0]}
    ) if "edge" in yield_type else None

    if expand_boundary_search:
        iterator = itertools.product(range(2), list(range(1, 2 * m)) + [0, 2 * m])
        ksymmetric_original = ksymmetric
    else:
        iterator = itertools.product(range(2), range(2 * m + 1))

    for u, w in iterator:
        # rail proposals preserve orientation in the target graph and only move in (w, k) quotient
        # graph.
        proposals = [(w, k) for k in range(t)]

        if expand_boundary_search:
            if w == 0:
                # b[1:3] is taken because those are the w and k indices
                proposals += [b[1:3] for b in _boundary_proposals(u, 1, tp, t, embedding)]
                ksymmetric = False
            elif w == 2 * m:
                proposals += [b[1:3] for b in _boundary_proposals(u, 2 * m - 1, tp, t, embedding)]
                ksymmetric = False
            else:
                ksymmetric = ksymmetric_original

        if ksymmetric or yield_type == "node":
            if yield_type in ("node", "rail-edge"):
                counts = [rail_score[(u, w_t, k_t)] for w_t, k_t in proposals]
            else:
                # the other only possibility is that yield_type == "edge". The following check is
                # just to avoid linter complaint about source_external_edges being possibly None.
                if source_external_edges is None:
                    raise ValueError("internal error: missing external edge subgraph")
                counts = [
                    rail_score[(u, w_t, k_t)] + sum(
                        int(target.has_edge(embedding[n_s], (u, w_t, k_t, j, z)))
                        for j in range(2)
                        for z in range(m)
                        # n_s will be nodes in the source graph with a different orientation
                        # to the current rail, that are neighbours of nodes in the current rail.
                        # Note that we pick k=0 because ksymmetric means that all k indices in the
                        # source rail are interchangeable, so we can just look at one of them.
                        for n_s in source_external_edges.neighbors((u, w, 0, j, z))
                        if n_s in embedding
                    )
                    for w_t, k_t in proposals
                ]

            p_indices = np.argpartition(np.asarray(counts), -tp)[-tp:]
            # Apply chosen rails to all nodes in the quotient rail block.
            embedding.update(
                {
                    (u, w, k, j, z): (u,) + proposals[p_indices[k]] + (j, z)
                    for k in range(tp)
                    for j in range(2)
                    for z in range(m)
                }
            )
        else:
            # this path is activated when ksymmetric is False and yield_type is either "edge" or
            # "rail-edge".
            if source_external_edges is None:
                raise ValueError("internal error: missing external edge subgraph")
            permutation_scores = {
                proposal_perm: sum(
                    rail_score[(u,) + proposal] for proposal in proposal_perm
                ) + sum(
                    int(target.has_edge(embedding[n_s], (u,) + proposal + (j, z)))
                    for k_s, proposal in enumerate(proposal_perm)
                    for j in range(2)
                    for z in range(m)
                    for n_s in source_external_edges.neighbors((u, w, k_s, j, z))
                    if n_s in embedding
                )
                for proposal_perm in itertools.permutations(proposals, tp)
            }
            selected = max(permutation_scores, key=lambda k: permutation_scores[k])
            embedding.update(
                {
                    (u, w, k, j, z): (u,) + selected[k] + (j, z)
                    for k in range(tp)
                    for j in range(2)
                    for z in range(m)
                }
            )

        if len(set(embedding.values())) != len(embedding):
            raise ValueError("Duplicate target coordinates detected in embedding")

    return embedding


def zephyr_quotient_search(
    source: nx.Graph,
    target: nx.Graph,
    *,
    quotient_search: QuotientSearchType = "by_quotient_rail",
    embedding: EmbeddingChain | None = None,
    expand_boundary_search: bool = True,
    ksymmetric: bool = False,
    yield_type: YieldType = "edge",
) -> tuple[EmbeddingChain, ZephyrSearchMetadata]:
    r"""Compute a high-yield Zephyr-to-Zephyr embedding.

    This routine starts from a source Zephyr graph with ``m`` rows and ``tp`` tiles,
    and maps it into a target Zephyr graph with the same ``m`` rows and ``t >= tp``
    tiles. It is designed for defective targets where a direct identity map may lose
    nodes or edges.

    The search is organised around the **quotient graph** of the Zephyr topology, formed by
    contracting fine-grained coordinate indices so that each equivalence class maps to a single
    quotient node. Two coarsenings are used:

    - **Quotient node** block :math:`(u, w, j, z)`: groups the ``tp`` source nodes that share
      orientation ``u``, column ``w``, intra-cell index ``j``, and row ``z`` but differ in
      tile index :math:`k \in \{0, \dots, tp-1\}`.
    - **Quotient rail** block :math:`(u, w)`: groups all :math:`2 m \cdot tp` nodes that share
      orientation ``u`` and column ``w`` (i.e. a whole Zephyr rail family) before any
      :math:`(k, j, z)` variation.

    The function can be used in (1) node-level mode (``quotient_search='by_quotient_node'``), where
    each quotient node block :math:`(u,w,j,z)` is optimised by choosing target candidates with the
    same :math:`(u,w,j,z)` and selecting the highest-yield proposals; (2) rail-level mode
    (``quotient_search='by_quotient_rail'``): optimise each quotient rail block :math:`(u,w,:)` by
    selecting rails :math:`(u,w_t,k_t)` that maximise yield.; and (3) hybrid mode
    (``quotient_search='by_rail_then_node'``): rail search followed by node refinement.

    When ``expand_boundary_search=True``, boundary columns ``w=0`` and ``w=2m`` are augmented using
    proposals drawn from adjacent internal columns. Whenever this behaviour is activated, nodes from
    the internal columns are assigned first, so that the unassigned nodes in the internal columns
    adjacent to the boundaries can be considered as proposals when optimising the boundary columns.

    Yield types control what the greedy search tries to preserve. ``"node"`` tries to place as
    many source nodes as possible onto target nodes that actually exist. ``"edge"`` tries to
    preserve source edges throughout the search. ``"rail-edge"`` is a mixed strategy: during rail
    search it first prefers rails that are internally well-formed, and if a node-refinement phase
    runs afterward it switches to ordinary edge-preservation scoring. The final yield for both
    ``"edge"`` and ``"rail-edge"`` is reported as a number of preserved source edges.

    Args:
        source (nx.Graph): Zephyr source graph (linear or coordinate labels).
        target (nx.Graph): Zephyr target graph (linear or coordinate labels).
        quotient_search (QuotientSearchType): Search strategy. One of ``'by_quotient_rail'``,
            ``'by_quotient_node'``, or ``'by_rail_then_node'``. See full docstrings for a
            description of these. Defaults to ``'by_quotient_rail'``.
        embedding (EmbeddingChain | None): Optional initial one-to-one chain mapping. If omitted,
            the identity on source coordinate indices is used (wrapped in singleton chains).
            Defaults to ``None``. This must be a chain mapping where each source node maps to
            a tuple of one or more target nodes (e.g., ``{source_node: (target_node,)}`` for
            singleton chains).
        expand_boundary_search (bool): Enable additional boundary proposals. Defaults to ``True``.
        ksymmetric (bool): Assume source ``k`` ordering can be treated symmetrically during greedy
            selection when valid. Defaults to ``False``.
        yield_type (YieldType): Optimization objective: ``'node'``, ``'edge'``, or ``'rail-edge'``.
            See full docstrings for a description of these. Defaults to ``'edge'``.

    Returns:
        tuple[EmbeddingChain, ZephyrSearchMetadata]:
            ``(embedding, metadata)``
            ``embedding`` is a pruned one-to-one chain embedding
            ``source_node -> (target_node,)`` (singleton chains). It contains only mappings whose
            target node exists in ``target``. Note that it is not guaranteed to preserve all source
            edges unless full edge-yield is achieved.
            ``metadata`` is a :class:`ZephyrSearchMetadata` namedtuple with fields
            ``max_num_yielded``, ``starting_num_yielded``, and ``final_num_yielded``.

    Note:
        If you want to refine a non-full-yield result with an external solver, run
        :func:`zephyr_quotient_search` first and only call the refinement routine when
        ``metadata.final_num_yielded < metadata.max_num_yielded``.

        .. code-block:: python

            emb, metadata = zephyr_quotient_search(source, target, yield_type="edge")
            if metadata.final_num_yielded < metadata.max_num_yielded:
                import minorminer

                initial_chains = {s: chain for s, chain in emb.items() if chain[0] in target}
                refined = minorminer.find_embedding(
                    S=source,
                    T=target,
                    initial_chains=initial_chains,
                    timeout=5,  # or whatever you want
                )
    """

    _validate_graph_inputs(source, target)
    m, tp, t = _extract_graph_properties(source, target)
    _validate_search_parameters(quotient_search, yield_type, embedding)

    _source, source_nodes, to_source = _ensure_coordinate_source(source, m, tp)
    _target, to_target = _ensure_coordinate_target(target, m, t)
    target_nodeset = set(_target.nodes())

    if embedding is None:
        working_embedding: Embedding = {n: n for n in source_nodes}
    else:
        # Convert chain format to internal single-node format
        working_embedding = {k: v[0] for k, v in embedding.items()}

    if yield_type == "node":
        max_num_yielded = source.number_of_nodes()
        num_yielded = sum(
            _target.has_node(working_embedding[n])
            for n in _source.nodes()
            if n in working_embedding
        )
    else:
        max_num_yielded = source.number_of_edges()
        num_yielded = sum(
            _target.has_edge(working_embedding[n1], working_embedding[n2])
            for n1, n2 in _source.edges()
            if n1 in working_embedding and n2 in working_embedding
        )

    full_yield = max_num_yielded == num_yielded
    starting_yield = num_yielded

    if not full_yield:
        supplement = quotient_search == "by_rail_then_node"

        if quotient_search == "by_quotient_rail" or supplement:
            working_embedding = _rail_search(
                source=_source,
                target=_target,
                embedding=working_embedding,
                # if quotient_search is by_rail_then_node, we expand boundary search only in the
                # node search, and disable it in the rail search:
                expand_boundary_search=((not supplement) and expand_boundary_search),
                ksymmetric=ksymmetric,
                yield_type=yield_type,
            )
            if supplement:
                working_embedding = _node_search(
                    source=_source,
                    target=_target,
                    embedding=working_embedding,
                    expand_boundary_search=expand_boundary_search,
                    ksymmetric=False,
                    yield_type=yield_type,
                )
        elif quotient_search == "by_quotient_node":
            working_embedding = _node_search(
                source=_source,
                target=_target,
                embedding=working_embedding,
                expand_boundary_search=expand_boundary_search,
                ksymmetric=ksymmetric,
                yield_type=yield_type,
            )

        if yield_type == "node":
            num_yielded = sum(
                _target.has_node(working_embedding[n])
                for n in _source.nodes()
                if n in working_embedding
            )
        else:
            num_yielded = sum(
                _target.has_edge(working_embedding[n1], working_embedding[n2])
                for n1, n2 in _source.edges()
                if n1 in working_embedding and n2 in working_embedding
            )
        full_yield = max_num_yielded == num_yielded

        if num_yielded < starting_yield:
            raise ValueError("Greedy quotient search reduced the objective value")

    

    pruned_embedding = {
        to_source(k): to_target(v)
        for k, v in working_embedding.items()
        if v in target_nodeset
    }  # TODO:?: why would a target node in the working_embedding not be in the target_nodeset?
    
    # Convert to chain format for return value
    pruned_embedding = {k: (v,) for k, v in pruned_embedding.items()}
    
    if full_yield and yield_type != "node":
        verify_embedding(emb=pruned_embedding, source=source, target=target)

    metadata = ZephyrSearchMetadata(
        max_num_yielded=max_num_yielded,
        starting_num_yielded=starting_yield,
        final_num_yielded=num_yielded,
    )
    return pruned_embedding, metadata


