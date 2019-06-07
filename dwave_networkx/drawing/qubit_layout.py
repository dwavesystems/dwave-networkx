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
#
# ================================================================================================
"""
Tools to visualize Chimera lattices and weighted graph problems on them.
"""

from __future__ import division

import networkx as nx
from networkx import draw

from dwave_networkx import _PY2

from dwave_networkx.drawing.distinguishable_colors import distinguishable_color_map

# compatibility for python 2/3
if _PY2:
    range = xrange

    def itervalues(d): return d.itervalues()

    def iteritems(d): return d.iteritems()
else:
    def itervalues(d): return d.values()

    def iteritems(d): return d.items()

__all__ = ['draw_qubit_graph']


def draw_qubit_graph(G, layout, linear_biases={}, quadratic_biases={},
                     nodelist=None, edgelist=None, cmap=None, edge_cmap=None, vmin=None, vmax=None,
                     edge_vmin=None, edge_vmax=None,
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

    linear_biases : dict (optional, default {})
        A dict of biases associated with each node in G. Should be of
        form {node: bias, ...}. Each bias should be numeric.

    quadratic_biases : dict (optional, default {})
        A dict of biases associated with each edge in G. Should be of
        form {edge: bias, ...}. Each bias should be numeric. Self-loop
        edges (i.e., :math:`i=j`) are treated as linear biases.

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

    if linear_biases or quadratic_biases:
        # if linear biases and/or quadratic biases are provided, then color accordingly.

        if nodelist is None:
            nodelist = G.nodes()

        if edgelist is None:
            edgelist = G.edges()

        if cmap is None:
            cmap = plt.get_cmap('coolwarm')

        if edge_cmap is None:
            edge_cmap = plt.get_cmap('coolwarm')

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

        kwargs['edge_color'] = edge_color
        kwargs['node_color'] = node_color

        # the range of the color map is shared for nodes/edges and is symmetric
        # around 0.
        vmag = max(max(abs(c) for c in node_color), max(abs(c) for c in edge_color))
        if vmin is None:
            vmin = -1 * vmag
        if vmax is None:
            vmax = vmag
        if edge_vmin is None:
            edge_vmin = -1 * vmag
        if edge_vmax is None:
            edge_vmax = vmag

    fig = plt.figure(1)
    ax = kwargs.pop('ax',None)
    cax = kwargs.pop('cax',None)
    if linear_biases or quadratic_biases:
        if ax is None:
            ax = fig.add_axes([0.01, 0.01, 0.86, 0.98])
        if cax is None:
            cax = fig.add_axes([.87, 0.2, 0.02, 0.6])  # left, bottom, width, height

    else:
        if ax is None:
            ax = fig.add_axes([0.01, 0.01, 0.98, 0.98])

    draw(G, layout, ax=ax, nodelist=nodelist, edgelist=edgelist,
         cmap=cmap, edge_cmap=edge_cmap, vmin=vmin, vmax=vmax, edge_vmin=edge_vmin,
         edge_vmax=edge_vmax,
         **kwargs)

    # if the biases are provided, then add a legend explaining the color map
    if linear_biases:
        mpl.colorbar.ColorbarBase(cax, cmap=cmap,
                                  norm=mpl.colors.Normalize(vmin=vmin, vmax=vmax, clip=False),
                                  orientation='vertical')

    if quadratic_biases:
        mpl.colorbar.ColorbarBase(cax, cmap=edge_cmap,
                                  norm=mpl.colors.Normalize(vmin=edge_vmin, vmax=edge_vmax, clip=False),
                                  orientation='vertical')



def draw_embedding(G, layout, emb, embedded_graph=None, interaction_edges=None,
                   chain_color=None, unused_color=(0.9,0.9,0.9,1.0), cmap=None,
                   show_labels=False, **kwargs):
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

    unused_color : tuple or color string (optional, default (0.9,0.9,0.9,1.0))
        The color to use for nodes and edges of G which are not involved
        in chains, and edges which are neither chain edges nor interactions.
        If unused_color is None, these nodes and edges will not be shown at all.

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

    if nx.utils.is_string_like(unused_color):
        from matplotlib.colors import colorConverter
        alpha = kwargs.get('alpha',1.0)
        unused_color = colorConverter.to_rgba(unused_color,alpha)

    if chain_color is None:
        import matplotlib.cm
        n = max(1., len(emb) - 1.)
        if kwargs.get("cmap"):
            color = matplotlib.cm.get_cmap(kwargs.get("cmap"))
        else:
            color = distinguishable_color_map(int(n+1))
        var_i = {v: i for i, v in enumerate(emb)}
        chain_color = {v: color(i/n) for i, v in enumerate(emb)}

    qlabel = {q: v for v, chain in iteritems(emb) for q in chain}
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

def draw_yield(G, layout, perfect_graph, unused_color=(0.9,0.9,0.9,1.0),
                    fault_color=(1.0,0.0,0.0,1.0), fault_shape='x',
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
        **kwargs )

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
