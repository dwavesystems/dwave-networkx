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
Tools to visualize :term:`Chimera` lattices and weighted :term:`graph` problems on them.
"""

import networkx as nx
from networkx import draw

from dwave_networkx.drawing.qubit_layout import draw_qubit_graph, draw_embedding, draw_yield
from dwave_networkx.generators.chimera import chimera_graph, find_chimera_indices, chimera_coordinates


__all__ = ['chimera_layout', 'draw_chimera', 'draw_chimera_embedding', 'draw_chimera_yield']


def chimera_layout(G, scale=1., center=None, dim=2):
    """Position the nodes of graph ``G`` in a Chimera cross topology.

    NumPy (https://scipy.org) is required for this function.

    Parameters
    ----------
    G : NetworkX graph
        :term:`Chimera` :term:`graph` or :term:`subgraph` of a
        Chimera graph. If every node in ``G`` has a ``chimera_index``
        attribute, the node position in the ``chimera_index``
        attribute is used to place each node. Otherwise,
        a best-effort attempt is made to find the node positions.

    scale : float (default 1.)
        Scale factor. If ``scale = 1``, then all positions fit within [0, 1]
        on the x-axis and [-1, 0] on the y-axis.

    center : None or array (default None)
        Coordinates of the top left corner.

    dim : int (default 2)
        Number of dimensions. If ``dim > 2``, then all extra dimensions are
        set to 0.

    Returns
    -------
    pos : dict
        Dictionary of positions keyed by node.

    Examples
    --------
    >>> G = dnx.chimera_graph(1)
    >>> pos = dnx.chimera_layout(G)

    """

    if not isinstance(G, nx.Graph):
        empty_graph = nx.Graph()
        empty_graph.add_edges_from(G)
        G = empty_graph

    # now we get chimera coordinates for the translation
    # first, check if we made it
    if G.graph.get("family") == "chimera":
        m = G.graph['rows']
        n = G.graph['columns']
        t = G.graph['tile']
        # get a node placement function
        xy_coords = chimera_node_placer_2d(m, n, t, scale, center, dim)

        if G.graph.get('labels') == 'coordinate':
            pos = {v: xy_coords(*v) for v in G.nodes()}
        elif G.graph.get('data'):
            pos = {v: xy_coords(*dat['chimera_index']) for v, dat in G.nodes(data=True)}
        else:
            coord = chimera_coordinates(m, n, t)
            pos = {v: xy_coords(*coord.linear_to_chimera(v)) for v in G.nodes()}
    else:
        # best case scenario, each node in G has a chimera_index attribute. Otherwise
        # we will try to determine it using the find_chimera_indices function.
        if all('chimera_index' in dat for __, dat in G.nodes(data=True)):
            chimera_indices = {v: dat['chimera_index'] for v, dat in G.nodes(data=True)}
        else:
            chimera_indices = find_chimera_indices(G)

        # we could read these off of the name attribute for G, but we would want the values in
        # the nodes to override the name in case of conflict.
        m = max(idx[0] for idx in chimera_indices.values()) + 1
        n = max(idx[1] for idx in chimera_indices.values()) + 1
        t = max(idx[3] for idx in chimera_indices.values()) + 1
        xy_coords = chimera_node_placer_2d(m, n, t, scale, center, dim)

        # compute our coordinates
        pos = {v: xy_coords(i, j, u, k) for v, (i, j, u, k) in chimera_indices.items()}

    return pos


def chimera_node_placer_2d(m, n, t, scale=1., center=None, dim=2):
    """Generate a function that converts Chimera indices to x- and
    y-coordinates for a plot.

    Parameters
    ----------
    m : int
        Number of rows in the :term:`Chimera` lattice.

    n : int
        Number of columns in the Chimera lattice.

    t : int
        Size of the shore within each Chimera tile.

    scale : float (default 1.)
        Scale factor. If ``scale = 1``, then all positions fit within [0, 1]
        on the x-axis and [-1, 0] on the y-axis.

    center : None or array (default None)
        Coordinates of the top left corner.

    dim : int (default 2)
        Number of dimensions. If ``dim > 2``, then all extra dimensions are
        set to 0.

    Returns
    -------
    xy_coords : function
        Function that maps a Chimera index ``(i, j, u, k)`` in an
        ``(m, n, t)`` Chimera lattice to x- and y-coordinates.

    """
    import numpy as np

    tile_center = t // 2
    tile_length = t + 3  # 1 for middle of cross, 2 for spacing between tiles
    # want the enter plot to fill in [0, 1] when scale=1
    scale /= max(m, n) * tile_length - 3

    grid_offsets = {}

    if center is None:
        center = np.zeros(dim)
    else:
        center = np.asarray(center)

    paddims = dim - 2
    if paddims < 0:
        raise ValueError("layout must have at least two dimensions")

    if len(center) != dim:
        raise ValueError("length of center coordinates must match dimension of layout")

    def _xy_coords(i, j, u, k):
        # row, col, shore, shore index

        # first get the coordinatiates within the tile
        if k < tile_center:
            p = k
        else:
            p = k + 1

        if u:
            xy = np.array([tile_center, -1 * p])
        else:
            xy = np.array([p, -1 * tile_center])

        # next offset the corrdinates based on the which tile
        if i > 0 or j > 0:
            if (i, j) in grid_offsets:
                xy += grid_offsets[(i, j)]
            else:
                off = np.array([j * tile_length, -1 * i * tile_length])
                xy += off
                grid_offsets[(i, j)] = off

        # convention for Chimera-lattice pictures is to invert the y-axis
        return np.hstack((xy * scale, np.zeros(paddims))) + center

    return _xy_coords


def draw_chimera(G, **kwargs):
    """Draw graph ``G`` in a Chimera cross topology.

    Linear and quadratic biases are visualized on the plot as specified
    in the ``linear_biases`` and ``quadratic_biases`` parameters.

    Parameters
    ----------
    G : NetworkX graph
        :term:`Chimera` :term:`graph` or a :term:`subgraph` of a Chimera graph.

    linear_biases : dict (optional, default {})
        A dict of biases associated with each node in ``G`` and of
        the form ``{node: bias, ...}``. Each bias is numeric.

    quadratic_biases : dict (optional, default {})
        A dict of biases associated with each edge in ``G`` and of
        the form ``{edge: bias, ...}``. Each bias is numeric. Self-loop
        edges (i.e., :math:`i=j`) are treated as linear biases.

    kwargs : optional keywords
       Parameters in :func:`~networkx.drawing.nx_pylab.draw_networkx`, except for the ``pos`` parameter.
       If the ``linear_biases`` or ``quadratic_biases`` parameters are specified,
       then the :func:`~networkx.drawing.nx_pylab.draw_networkx` ``node_color``
       or ``edge_color`` parameters are ignored.

    Examples
    --------
    >>> # Plot 2x2 Chimera unit cells
    >>> import networkx as nx
    >>> import dwave_networkx as dnx
    >>> import matplotlib.pyplot as plt  # doctest: +SKIP
    >>> G = dnx.chimera_graph(2, 2, 4)
    >>> dnx.draw_chimera(G)  # doctest: +SKIP
    >>> plt.show()  # doctest: +SKIP

    """
    draw_qubit_graph(G, chimera_layout(G), **kwargs)


def draw_chimera_embedding(G, *args, **kwargs):
    """Draw an embedding onto the chimera graph ``G``, according to the Chimera layout.

    If the ``interaction_edges`` parameter is not None, then only display the couplers in that
    list. If the ``embedded_graph`` parameter is not None, then only display the couplers between
    chains with intended couplings according to the ``embedded_graph`` parameter.

    Parameters
    ----------
    G : NetworkX graph
        :term:`Chimera` :term:`graph` or a :term:`subgraph` of a Chimera graph.

    emb : dict
        A dict of chains associated with each node in ``G`` and
        of the form ``{node: chain, ...}``. Chains are iterables
        of qubit labels (qubits are nodes in ``G``).

    embedded_graph : NetworkX graph (optional, default None)
        Graph which contains all keys of the ``emb`` parameter as nodes. If specified, then
        the edges of ``G`` will be considered interactions if and only if they
        exist between two chains of the ``emb`` parameter and if their keys are connected by
        an edge in the ``embedded_graph`` parameter.

    interaction_edges : list (optional, default None)
        List of edges which will be used as interactions.

    show_labels: boolean (optional, default False)
        If True, then each chain in the ``emb`` parameter is labelled with its key.

    chain_color : dict (optional, default None)
        A dict of colors associated with each key in the ``emb`` parameter and
        of the form ``{node: rgba_color, ...}``. Colors must be length-4
        tuples of floats between 0 and 1, inclusive. If None,
        then each chain will be assigned a different color.

    unused_color : tuple (optional, default (0.9,0.9,0.9,1.0))
        Color to use for graph ``G``'s nodes that are not part of
        chains and edges that are neither chain edges nor interactions.
        If None, then these nodes and edges will not be shown.

    overlapped_embedding: boolean (optional, default False)
        If True, then chains in the ``emb`` parameter may overlap (contain
        the same vertices in ``G``), and the drawing will display these overlaps as
        concentric circles.

    kwargs : optional keywords
       Parameters in :func:`~networkx.drawing.nx_pylab.draw_networkx`, except for the ``pos`` parameter.
       If the ``linear_biases`` or ``quadratic_biases`` parameters are specified,
       then the :func:`~networkx.drawing.nx_pylab.draw_networkx` ``node_color``
       or ``edge_color`` parameters are ignored.
    """
    draw_embedding(G, chimera_layout(G), *args, **kwargs)


def draw_chimera_yield(G, **kwargs):
    """Draw the given graph ``G`` with highlighted faults, according to the Chimera layout.

    Parameters
    ----------
    G : NetworkX graph
        :term:`Graph` to be parsed for faults.

    unused_color : tuple or color string (optional, default (0.9,0.9,0.9,1.0))
        Color to use for graph ``G``'s nodes and edges which are not faults.
        If None, then these nodes and edges will not be shown.

    fault_color : tuple or color string (optional, default (1.0,0.0,0.0,1.0))
        Color to represent nodes that are absent from graph ``G``. Colors must be
        length-4 tuples of floats between 0 and 1, inclusive.

    fault_shape : string, optional (default='x')
        Shape of the fault nodes. The shapes are the same as those specified for
        `Matplotlib markers <https://matplotlib.org/stable/api/markers_api.html#module-matplotlib.markers>`_.

    fault_style : string, optional (default='dashed')
        Line style for fault edges. The line style can be any of the following values:
        ``'solid'``, ``'dashed'``, ``'dotted'``, ``'dashdot'``.

    kwargs : optional keywords
       Parameters in :func:`~networkx.drawing.nx_pylab.draw_networkx`, except for the ``pos`` parameter.
       If the ``linear_biases`` or ``quadratic_biases`` parameters are specified,
       then the :func:`~networkx.drawing.nx_pylab.draw_networkx` ``node_color``
       or ``edge_color`` parameters are ignored.
    """
    try:
        assert(G.graph["family"] == "chimera")
        m = G.graph["rows"]
        n = G.graph["columns"]
        t = G.graph["tile"]
        coordinates = G.graph["labels"] == "coordinate"
    except:
        raise ValueError("Target chimera graph needs to have columns, rows, \
        tile, and label attributes to be able to identify faulty qubits.")

    perfect_graph = chimera_graph(m,n,t, coordinates=coordinates)

    draw_yield(G, chimera_layout(perfect_graph), perfect_graph, **kwargs)
