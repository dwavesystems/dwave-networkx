# Copyright 2024 D-Wave
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

from typing import Optional
import math

import networkx as nx
import numpy

from dwave.graphs.drawing.chimera_layout import draw_chimera, chimera_layout
from dwave.graphs.drawing.pegasus_layout import draw_pegasus, pegasus_layout
from dwave.graphs.drawing.zephyr_layout import draw_zephyr, zephyr_layout

__all__ = ["draw_parallel_embeddings"]


def _generate_node_color_dict(
    G: nx.Graph,
    embeddings: list[dict],
    S: Optional[nx.Graph] = None,
    one_to_iterable: bool = False,
    shuffle_colormap: bool = True,
    seed: Optional[int] = None,
) -> tuple[dict, list[dict]]:
    """Generate a node color dictionary mapping each node in G to an embedding index or NaN.

    Args:
        G: The target graph.
        embeddings: A list of embeddings (each embedding a dict from source nodes to target nodes).
        S: The optional source graph.
        one_to_iterable: If True, a single source node maps to multiple target nodes.
        shuffle_colormap: If True, embeddings are shuffled before assigning colors.
        seed: A seed for the random number generator. Only used when ``shuffle_colormap`` is True.

    Returns:
        node_color_dict: A dictionary mapping each node in G to either an embedding index or NaN.
        embeddings_: The potentially shuffled embeddings list used for assigning colors.
    """
    node_color_dict = {q: float("nan") for q in G.nodes()}

    if shuffle_colormap:
        embeddings_ = embeddings.copy()
        rng = numpy.random.default_rng(seed=seed)
        rng.shuffle(embeddings_)
    else:
        embeddings_ = embeddings

    if S is None:
        # If there is no source graph, color all nodes in the embeddings
        if one_to_iterable:
            # Multiple target nodes per source node
            node_color_dict.update(
                {
                    q: idx
                    for idx, emb in enumerate(embeddings_)
                    for c in emb.values()
                    for q in c
                }
            )
        else:
            # One-to-one mapping
            node_color_dict.update(
                {q: idx for idx, emb in enumerate(embeddings_) for q in emb.values()}
            )
    else:
        # If a source graph is provided, only color nodes corresponding to S
        node_set = set(S.nodes())
        if one_to_iterable:
            node_color_dict.update(
                {
                    q: idx if n in node_set else float("nan")
                    for idx, emb in enumerate(embeddings_)
                    for n, c in emb.items()
                    for q in c
                }
            )
        else:
            node_color_dict.update(
                {
                    q: idx
                    for idx, emb in enumerate(embeddings_)
                    for n, q in emb.items()
                    if n in node_set
                }
            )

    return node_color_dict, embeddings_


def _generate_edge_color_dict(
    G: nx.Graph,
    embeddings: list[dict],
    S: Optional[nx.Graph],
    one_to_iterable: bool,
    node_color_dict: dict,
) -> dict:
    """Generate an edge color dictionary mapping each edge in G to an embedding index or NaN.

    Args:
        G: The target graph.
        embeddings: A list of embeddings (each embedding a dict from source nodes to target nodes).
        S: The optional source graph (if None, edges are colored based on node colors).
        one_to_iterable: If True, a single source node maps to multiple target nodes.
        node_color_dict: The node color dictionary to reference for edge coloring.

    Returns:
        edge_color_dict: A dictionary mapping each edge in G to either an embedding index or NaN.
    """
    if S is not None:
        # Edges corresponding to source graph embeddings
        edge_color_dict = {
            (tu, tv): idx
            for idx, emb in enumerate(embeddings)
            for u, v in S.edges()
            if u in emb and v in emb
            for tu in (emb[u] if one_to_iterable else [emb[u]])
            for tv in (emb[v] if one_to_iterable else [emb[v]])
            if G.has_edge(tu, tv)
        }

        if one_to_iterable:
            # Add chain edges
            for idx, emb in enumerate(embeddings):
                for chain in emb.values():
                    Gchain = G.subgraph(chain)
                    edge_color_dict.update({e: idx for e in Gchain.edges()})
    else:
        # If no source graph, color edges where both endpoints share the same embedding color
        edge_color_dict = {
            (v1, v2): node_color_dict[v1]
            for v1, v2 in G.edges()
            if node_color_dict[v1] == node_color_dict[v2]
        }

    return edge_color_dict


def draw_parallel_embeddings(
    G: nx.Graph,
    embeddings: list[dict],
    S: Optional[nx.Graph] = None,
    one_to_iterable: bool = True,
    shuffle_colormap: bool = True,
    seed: Optional[int] = None,
    use_plt: bool = True,
    **kwargs,
):
    """Visualizes the embeddings using dwave-graphs layout functions.

    Args:
        G: The target graph to be visualized.
        embeddings: A list of embeddings, each embedding is a dictionary. A
           list of embeddings can be created using methods within the module
           `minorminer.utils.parallel_embeddings`
        S: The source graph to visualize. If provided then only
           edges relevant to the source graph are shown, as opposed to all
           those supported by the embedding.
        one_to_iterable: If True, allow multiple target nodes per source node,
            values of the embedding are iterables on nodes of G.
            If False, embedding values should be nodes of G.
        shuffle_colormap: If True, randomize the colormap assignment.
        seed: A seed for the random number generator. Only used when ``shuffle_colormap`` is True.
        **kwargs: Additional keyword arguments for the drawing functions.

    Examples
    --------
    This example plots 3 embeddings of a length-8 path on Chimera[m=2], with
    no embedding in the bottom right.

    >>> import networkx as nx
    >>> import dwave.graphs
    >>> import matplotlib.pyplot as plt   # doctest: +SKIP
    >>> G = dwave.graphs.chimera_graph(2)
    >>> S = nx.from_edgelist({(i,i+1) for i in range(7)})
    >>> emb = {i: (i // 2) + 4*(i % 2) for i in range(8)} # Top-left embedding
    >>> embs = [{k: (v+8*offset,) for k,v in emb.items()} for offset in range(3)]
    >>> dwave.graphs.draw_parallel_embeddings(G, embs)    # doctest: +SKIP
    >>> plt.show()    # doctest: +SKIP
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("Matplotlib is required for draw_parallel_embeddings()")

    ax = plt.gca()
    cmap = plt.get_cmap("turbo").copy()
    cmap.set_bad("lightgrey")

    # Generate node_color_dict and embeddings to use
    node_color_dict, _embeddings = _generate_node_color_dict(
        G, embeddings, S, one_to_iterable, shuffle_colormap, seed
    )

    # Generate edge_color_dict
    edge_color_dict = _generate_edge_color_dict(
        G, _embeddings, S, one_to_iterable, node_color_dict
    )

    # Default drawing arguments
    draw_kwargs = {
        "G": G,
        "node_color": [node_color_dict[q] for q in G.nodes()],
        "edge_color": "lightgrey",
        "node_shape": "o",
        "ax": ax,
        "with_labels": False,
        "width": 1,
        "cmap": cmap,
        "edge_cmap": cmap,
        "node_size": 300 / math.sqrt(G.number_of_nodes()),
    }
    draw_kwargs.update(kwargs)

    topology = G.graph.get("family")
    # Draw the combined graph with color mappings
    if topology == "chimera":
        pos = chimera_layout(G)
        draw_chimera(**draw_kwargs)
    elif topology == "pegasus":
        pos = pegasus_layout(G)
        draw_pegasus(**draw_kwargs)
    elif topology == "zephyr":
        pos = zephyr_layout(G)
        draw_zephyr(**draw_kwargs)
    else:
        pos = nx.spring_layout(G)
        nx.draw_networkx(**draw_kwargs)

    if len(edge_color_dict) > 0:
        # Recolor specific edges on top of the original graph
        nx.draw_networkx_edges(
            G,
            pos=pos,
            edgelist=list(edge_color_dict.keys()),
            edge_color=list(edge_color_dict.values()),
            width=1,
            edge_cmap=cmap,
            ax=ax,
        )
