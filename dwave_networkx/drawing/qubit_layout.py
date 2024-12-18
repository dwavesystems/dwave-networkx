# Copyright 2018 D-Wave Systems Inc.
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

"""
Tools to visualize Chimera lattices and weighted graph problems on them.
"""

import math
import random
import networkx as nx

from networkx import draw

from dwave_networkx.drawing.distinguishable_colors import distinguishable_color_map

# new imports added to supports visualize parallel embedding function
import numpy as np
from typing import Union, List, Dict, Tuple, Optional
import matplotlib as plt
from dwave_networkx.drawing.chimera_layout import draw_chimera, chimera_layout
from dwave_networkx.drawing.pegasus_layout import draw_pegasus, pegasus_layout
from dwave_networkx.drawing.zephyr_layout import draw_zephyr, zephyr_layout

__all__ = ['draw_qubit_graph']


def draw_qubit_graph(G, layout, linear_biases=None, quadratic_biases=None,
                     nodelist=None, edgelist=None, midpoint=None,
                     **kwargs):
    """Draws graph G according to layout.

    If `linear_biases` and/or `quadratic_biases` are provided, these
    are visualized on the plot.

    Parameters
    ----------
    G : NetworkX graph
        The graph to be drawn

    layout : dict
        A dict of coordinates associated with each node in G.  Should
        be of the form {node: coordinate, ...}.  Coordinates will be
        treated as vectors, and should all have the same length.

    linear_biases : dict (optional, None)
        A dict of biases associated with each node in G. Should be of
        form {node: bias, ...}. Each bias should be numeric.

    quadratic_biases : dict (optional, None)
        A dict of biases associated with each edge in G. Should be of
        form {edge: bias, ...}. Each bias should be numeric. Self-loop
        edges (i.e., :math:`i=j`) are treated as linear biases.

    midpoint : float (optional, default None)
        A float that specifies where the center of the colormap should
        be. If not provided, the colormap will default to the middle of
        min/max values provided.

    kwargs : optional keywords
       See networkx.draw_networkx() for a description of optional keywords,
       with the exception of the `pos` parameter which is not used by this
       function. If `linear_biases` or `quadratic_biases` are provided,
       any provided `node_color` or `edge_color` arguments are ignored.

    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib as mpl
    except ImportError:
        raise ImportError("Matplotlib and numpy required for draw_qubit_graph()")

    try:
        from mpl_toolkits.axes_grid1 import make_axes_locatable
    except ImportError:
        _mpl_toolkit_found = False
    else:
        _mpl_toolkit_found = True

    linear_biases = linear_biases or dict()
    quadratic_biases = quadratic_biases or dict()

    fig = plt.gcf()
    ax = kwargs.pop('ax', None)
    cax = kwargs.pop('cax', None)

    if linear_biases or quadratic_biases:
        # if linear biases and/or quadratic biases are provided, then color accordingly.

        if ax is None:
            ax = fig.add_axes([0.01, 0.01, 0.86, 0.98])

        if cax is None:
            if _mpl_toolkit_found:
                divider = make_axes_locatable(ax)
                cax = divider.append_axes('right', size='2%', pad=0.05)
            else:
                cax = fig.add_axes([.87, 0.2, 0.02, 0.6])  # left, bottom, width, height

        if nodelist is None:
            nodelist = G.nodes()

        if edgelist is None:
            edgelist = G.edges()

        # any edges or nodes with an unspecified bias default to 0
        def edge_color(u, v):
            c = 0.
            if (u, v) in quadratic_biases:
                c += quadratic_biases[(u, v)]
            if (v, u) in quadratic_biases:
                c += quadratic_biases[(v, u)]
            return c

        def node_color(v):
            c = 0.
            if v in linear_biases:
                c += linear_biases[v]
            if (v, v) in quadratic_biases:
                c += quadratic_biases[(v, v)]
            return c

        node_color = [node_color(v) for v in nodelist]
        edge_color = [edge_color(u, v) for u, v in edgelist]

        # the range of the color map is shared for nodes/edges and is symmetric
        # around 0.
        vmag = max(max(abs(c) for c in node_color), max(abs(c) for c in edge_color))

        # since we're applying the colormap here, matplotlib throws warnings if
        # we provide these arguments and it doesn't use them.
        cmap = kwargs.pop('cmap', None)
        cmap = plt.get_cmap('coolwarm') if cmap is None else cmap
        vmin = kwargs.pop('vmin', None)
        vmin = -vmag if vmin is None else vmin
        vmax = kwargs.pop('vmax', None)
        vmax = vmag if vmax is None else vmax

        edge_cmap = kwargs.pop('edge_cmap', None)
        edge_cmap = plt.get_cmap('coolwarm') if edge_cmap is None else edge_cmap
        edge_vmin = kwargs.pop('edge_vmin', None)
        edge_vmin = -vmag if edge_vmin is None else edge_vmin
        edge_vmax = kwargs.pop('edge_vmax', None)
        edge_vmax = vmag if edge_vmax is None else edge_vmax

        if linear_biases and quadratic_biases:
            final_vmin = min(edge_vmin, vmin)
            final_vmax = max(edge_vmax, vmax)
            mapper = cmap

        elif linear_biases:
            final_vmin = vmin
            final_vmax = vmax
            mapper = cmap

        elif quadratic_biases:
            final_vmin = edge_vmin
            final_vmax = edge_vmax
            mapper = edge_cmap

        midpoint = midpoint or (final_vmax + final_vmin) / 2.0
        norm_map = mpl.colors.TwoSlopeNorm(midpoint, vmin=final_vmin, vmax=final_vmax)
        mpl.colorbar.ColorbarBase(cax, cmap=mapper, norm=norm_map, orientation='vertical')
        kwargs['node_color'] = [mapper(norm_map(node)) for node in node_color]
        kwargs['edge_color'] = [mapper(norm_map(edge)) for edge in edge_color]

    else:
        if ax is None:
            ax = fig.add_axes([0.01, 0.01, 0.98, 0.98])

    draw(G, layout, ax=ax, nodelist=nodelist, edgelist=edgelist, **kwargs)


def draw_embedding(G, layout, emb, embedded_graph=None, interaction_edges=None,
                   chain_color=None, unused_color=(0.9, 0.9, 0.9, 1.0), cmap=None,
                   show_labels=False, overlapped_embedding=False, **kwargs):
    """Draws an embedding onto the graph G, according to layout.

    If interaction_edges is not None, then only display the couplers in that
    list.  If embedded_graph is not None, the only display the couplers between
    chains with intended couplings according to embedded_graph.

    Parameters
    ----------
    G : NetworkX graph
        The graph to be drawn

    layout : dict
        A dict of coordinates associated with each node in G.  Should
        be of the form {node: coordinate, ...}.  Coordinates will be
        treated as vectors, and should all have the same length.

    emb : dict
        A dict of chains associated with each node in G.  Should be
        of the form {node: chain, ...}.  Chains should be iterables
        of qubit labels (qubits are nodes in G).

    embedded_graph : NetworkX graph (optional, default None)
        A graph which contains all keys of emb as nodes.  If specified,
        edges of G will be considered interactions if and only if they
        exist between two chains of emb if their keys are connected by
        an edge in embedded_graph

    interaction_edges : list (optional, default None)
        A list of edges which will be used as interactions.

    show_labels: boolean (optional, default False)
        If show_labels is True, then each chain in emb is labelled with its key.

    chain_color : dict (optional, default None)
        A dict of colors associated with each key in emb.  Should be
        of the form {node: rgba_color, ...}.  Colors should be length-4
        tuples of floats between 0 and 1 inclusive. If chain_color is None,
        each chain will be assigned a different color.

    cmap : str or matplotlib colormap (optional, default None)
        A matplotlib colormap for coloring of chains.  Only used if chain_color
        is None.

    unused_color : tuple or color string (optional, default (0.9,0.9,0.9,1.0))
        The color to use for nodes and edges of G which are not involved
        in chains, and edges which are neither chain edges nor interactions.
        If unused_color is None, these nodes and edges will not be shown at all.

    overlapped_embedding: boolean (optional, default False)
        If overlapped_embedding is True, then chains in emb may overlap (contain
        the same vertices in G), and the drawing will display these overlaps as
        concentric circles.

    kwargs : optional keywords
       See networkx.draw_networkx() for a description of optional keywords,
       with the exception of the `pos` parameter which is not used by this
       function. If `linear_biases` or `quadratic_biases` are provided,
       any provided `node_color` or `edge_color` arguments are ignored.
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib as mpl
    except ImportError:
        raise ImportError("Matplotlib and numpy required for draw_chimera()")

    if isinstance(unused_color, str):
        from matplotlib.colors import colorConverter
        alpha = kwargs.get('alpha', 1.0)
        unused_color = colorConverter.to_rgba(unused_color, alpha)

    if chain_color is None:
        import matplotlib.cm
        n = max(1., len(emb) - 1.)
        if cmap:
            color = matplotlib.cm.get_cmap(cmap)
        else:
            color = distinguishable_color_map(int(n+1))
        var_i = {v: i for i, v in enumerate(emb)}
        chain_color = {v: color(i/n) for i, v in enumerate(emb)}

    if overlapped_embedding:
        bags = compute_bags(G, emb)
        base_node_size = kwargs.get('node_size', 100)
        node_size_dict = {v: base_node_size for v in G.nodes()}
        G, emb, interaction_edges = unoverlapped_embedding(G, emb, interaction_edges)
        for node, data in G.nodes(data=True):
            if 'dummy' in data:
                v, x = node
                layout[node] = layout[v]

        for v, bag in bags.items():
            for i, x in enumerate(bag):
                node_size_dict[(v, x)] = base_node_size * (len(bag) - i) ** 2

        kwargs['node_size'] = [node_size_dict[p] for p in G.nodes()]

    qlabel = {q: v for v, chain in emb.items() for q in chain}
    edgelist = []
    edge_color = []
    background_edgelist = []
    background_edge_color = []

    if interaction_edges is not None:
        interactions = nx.Graph()
        interactions.add_edges_from(interaction_edges)

        def show(p, q, u, v): return interactions.has_edge(p, q)
    elif embedded_graph is not None:
        def show(p, q, u, v): return embedded_graph.has_edge(u, v)
    else:
        def show(p, q, u, v): return True

    for (p, q) in G.edges():
        u = qlabel.get(p)
        v = qlabel.get(q)
        if u is None or v is None:
            ec = unused_color
        elif u == v:
            ec = chain_color.get(u)
        elif show(p, q, u, v):
            ec = (0, 0, 0, 1)
        else:
            ec = unused_color

        if ec == unused_color:
            background_edgelist.append((p, q))
            background_edge_color.append(ec)
        elif ec is not None:
            edgelist.append((p, q))
            edge_color.append(ec)

    nodelist = []
    node_color = []
    for p in G.nodes():
        u = qlabel.get(p)
        if u is None:
            pc = unused_color
        else:
            pc = chain_color.get(u)

        if pc is not None:
            nodelist.append(p)
            node_color.append(pc)

    labels = {}
    if show_labels:
        if overlapped_embedding:
            node_labels = {q: [] for q in bags.keys()}
            node_index = {p: i for i, p in enumerate(G.nodes())}
            for v in emb.keys():
                v_labelled = False
                chain = emb[v]
                for node in chain:
                    (q, _) = node
                    if len(bags[q]) == 1:
                        # if there's a node that only has this label, use that
                        labels[q] = str(v)
                        v_labelled = True
                        break
                if not v_labelled and chain:
                    # otherwise, pick a random node for this label
                    node = random.choice(list(chain))
                    (q, _) = node
                    node_labels[q].append(v)
            for q, label_vars in node_labels.items():
                x, y = layout[q]
                # TODO: find a better way of placing labels around the outside of nodes.
                # Currently, if the graph is resized, labels will appear at a strange distance from the vertices.
                # To fix this, the "scale" value below, rather than being a fixed constant, should be determined using
                # both the size of the nodes and the size of the coordinate space of the graph.
                scale = 0.1
                # spread the labels evenly around the node.
                for i, v in enumerate(label_vars):
                    theta = 2 * math.pi * i / len(label_vars)
                    new_x = x + scale * math.sin(theta)
                    new_y = y + scale * math.cos(theta)

                    plt.text(new_x, new_y, str(v), color=node_color[node_index[(q, v)]], horizontalalignment='center',
                             verticalalignment='center')
        else:
            for v in emb.keys():
                c = emb[v]
                labels[list(c)[0]] = str(v)

    # draw the background (unused) graph first
    if unused_color is not None:
        draw(G, layout, nodelist=nodelist, edgelist=background_edgelist,
             node_color=node_color, edge_color=background_edge_color,
             **kwargs)

    draw(G, layout, nodelist=nodelist, edgelist=edgelist,
         node_color=node_color, edge_color=edge_color, labels=labels,
         **kwargs)


def generate_node_color_dict(G: nx.Graph, embeddings: List[dict], S: nx.Graph = None,
    one_to_iterable: bool = False, shuffle_colormap: bool = True,
    seed: Union[int, np.random.RandomState, np.random.Generator] = None,) -> Tuple[Dict, List[dict]]:
    """Generate a node color dictionary mapping each node in G to an embedding index or NaN.
    
    Args:
        G: The target graph.
        embeddings: A list of embeddings (each embedding a dict from source nodes to target nodes).
        S: The optional source graph.
        one_to_iterable: If True, a single source node maps to multiple target nodes.
        shuffle_colormap: If True, embeddings are shuffled before assigning colors.
        seed: A seed for the random number generator.
    
    Returns:
        node_color_dict: A dictionary mapping each node in G to either an embedding index or NaN.
        _embeddings: The potentially shuffled embeddings list used for assigning colors.
    """
    node_color_dict = {q: float("nan") for q in G.nodes()}

    if shuffle_colormap:
        _embeddings = embeddings.copy()
        prng = np.random.default_rng(seed)
        prng.shuffle(_embeddings)
    else:
        _embeddings = embeddings

    if S is None:
        # If there is no source graph, color all nodes in the embeddings
        if one_to_iterable:
            # Multiple target nodes per source node
            node_color_dict.update(
                {
                    q: idx
                    for idx, emb in enumerate(_embeddings)
                    for c in emb.values()
                    for q in c
                }
            )
        else:
            # One-to-one mapping
            node_color_dict.update(
                {q: idx for idx, emb in enumerate(_embeddings) for q in emb.values()}
            )
    else:
        # If a source graph is provided, only color nodes corresponding to S
        node_set = set(S.nodes())
        if one_to_iterable:
            node_color_dict.update(
                {
                    q: idx if n in node_set else float("nan")
                    for idx, emb in enumerate(_embeddings)
                    for n, c in emb.items()
                    for q in c
                }
            )
        else:
            node_color_dict.update(
                {
                    q: idx
                    for idx, emb in enumerate(_embeddings)
                    for n, q in emb.items()
                    if n in node_set
                }
            )

    return node_color_dict, _embeddings


def generate_edge_color_dict(G: nx.Graph, embeddings: List[dict], S: nx.Graph,
    one_to_iterable: bool, node_color_dict: Dict) -> Dict:
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


def visualize_parallel_embeddings(G: nx.Graph, embeddings: List[dict], S: nx.Graph = None,
    one_to_iterable: bool = False, shuffle_colormap: bool = True,
    seed: Union[int, np.random.RandomState, np.random.Generator] = None, **kwargs,
) -> Tuple[dict, dict]:
    """Visualizes the embeddings using dwave_networkx's layout functions.

    Args:
        G: The target graph to be visualized.
        embeddings: A list of embeddings.
        S: The source graph to visualize (optional).
        one_to_iterable: If True, allow multiple target nodes per source node.
        shuffle_colormap: If True, randomize the colormap assignment.
        seed: A seed for the random number generator.
        **kwargs: Additional keyword arguments for the drawing functions.
    """
    ax = plt.gca()
    cmap = plt.get_cmap("turbo").copy()
    cmap.set_bad("lightgrey")

    # Generate node_color_dict and embeddings to use
    node_color_dict, _embeddings = generate_node_color_dict(
        G, embeddings, S, one_to_iterable, shuffle_colormap, seed
    )

    # Generate edge_color_dict
    edge_color_dict = generate_edge_color_dict(
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
        "node_size": 300 / np.sqrt(G.number_of_nodes()),
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


def compute_bags(C, emb):
    # Given an overlapped embedding, compute the set of source nodes embedded at every target node.
    bags = {v: [] for v in C.nodes()}
    for x, chain in emb.items():
        for v in chain:
            bags[v].append(x)
    return bags


def unoverlapped_embedding(G, emb, interaction_edges):
    # Given an overlapped embedding, construct a new graph and embedding without overlaps
    # by making copies of nodes that have multiple variables.

    bags = compute_bags(G, emb)
    new_G = G.copy()
    new_emb = dict()

    for x, chain in emb.items():
        for v in chain:
            new_G.add_node((v, x), dummy=True)
        for (u, v) in G.subgraph(chain).edges():
            new_G.add_edge((u, x), (v, x))
        new_emb[x] = {(v, x) for v in chain}

    for (u, v) in G.edges():
        for x in bags[u]:
            for y in bags[v]:
                new_G.add_edge((u, x), (v, y))

    if interaction_edges:
        new_interaction_edges = list(interaction_edges)
        for (u, v) in interaction_edges:
            for x in bags[u]:
                for y in bags[v]:
                    new_interaction_edges.append(((u, x), (v, y)))
    else:
        new_interaction_edges = None

    return new_G, new_emb, new_interaction_edges


def draw_yield(G, layout, perfect_graph, unused_color=(0.9, 0.9, 0.9, 1.0),
               fault_color=(1.0, 0.0, 0.0, 1.0), fault_shape='x',
               fault_style='dashed', **kwargs):
    """Draws the given graph G with highlighted faults, according to layout.

    Parameters
    ----------
    G : NetworkX graph
        The graph to be parsed for faults

    layout : dict
        A dict of coordinates associated with each node in perfect_graph. Should
        be of the form {node: coordinate, ...}.  Coordinates will be
        treated as vectors, and should all have the same length.

    perfect_graph : NetworkX graph
        The graph to be drawn with highlighted faults


    unused_color : tuple or color string (optional, default (0.9,0.9,0.9,1.0))
        The color to use for nodes and edges of G which are not faults.
        If unused_color is None, these nodes and edges will not be shown at all.

    fault_color : tuple or color string (optional, default (1.0,0.0,0.0,1.0))
        A color to represent nodes absent from the graph G. Colors should be
        length-4 tuples of floats between 0 and 1 inclusive.

    fault_shape : string, optional (default='x')
        The shape of the fault nodes. Specification is as matplotlib.scatter
        marker, one of 'so^>v<dph8'.

    fault_style : string, optional (default='dashed')
        Edge fault line style (solid|dashed|dotted,dashdot)

    kwargs : optional keywords
       See networkx.draw_networkx() for a description of optional keywords,
       with the exception of the `pos` parameter which is not used by this
       function. If `linear_biases` or `quadratic_biases` are provided,
       any provided `node_color` or `edge_color` arguments are ignored.
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib as mpl
    except ImportError:
        raise ImportError("Matplotlib and numpy required for draw_chimera()")

    nodelist = G.nodes()
    edgelist = G.edges()
    faults_nodelist = perfect_graph.nodes() - nodelist
    faults_edgelist = perfect_graph.edges() - edgelist

    # To avoid matplotlib.pyplot.scatter warnings for single tuples, create
    # lists of colors from given colors.
    faults_node_color = [fault_color for v in faults_nodelist]
    faults_edge_color = [fault_color for v in faults_edgelist]

    # Draw faults with different style and shape
    draw(perfect_graph, layout, nodelist=faults_nodelist, edgelist=faults_edgelist,
         node_color=faults_node_color, edge_color=faults_edge_color,
         style=fault_style, node_shape=fault_shape,
         **kwargs)

    # Draw rest of graph
    if unused_color is not None:
        if nodelist is None:
            nodelist = G.nodes() - faults_nodelist
        if edgelist is None:
            edgelist = G.edges() - faults_edgelist

        unused_node_color = [unused_color for v in nodelist]
        unused_edge_color = [unused_color for v in edgelist]

        draw(perfect_graph, layout, nodelist=nodelist, edgelist=edgelist,
             node_color=unused_node_color, edge_color=unused_edge_color,
             **kwargs)
