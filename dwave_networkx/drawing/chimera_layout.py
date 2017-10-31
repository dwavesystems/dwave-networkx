"""
Tools to visualize Chimera lattices and weighted graph problems on them.
"""

from __future__ import division

import networkx as nx
from networkx import draw

from dwave_networkx import _PY2
from dwave_networkx.generators.chimera import find_chimera_indices


# compatibility for python 2/3
if _PY2:
    range = xrange
    itervalues = lambda d: d.itervalues()
    iteritems = lambda d: d.iteritems()
else:
    itervalues = lambda d: d.values()
    iteritems = lambda d: d.items()

__all__ = ['chimera_layout', 'draw_chimera']


def chimera_layout(G, scale=1., center=None, dim=2):
    """Positions the nodes in a Chimera lattice.

    NumPy (http://scipy.org) is required for this function.

    Parameters
    ----------
    G : graph
        A networkx graph. Should be a Chimera graph or a subgraph of a
        Chimera graph. If every node in G has a 'chimera_index'
        attribute, then those are used to place the nodes. Otherwise will
        attempt to find positions, but is not guarunteed to succeed.

    scale : float (default 1.)
        Scale factor. When scale = 1 the all positions will fit within [0, 1]
        on the x-axis and [-1, 0] on the y-axis.

    center : None or array (default None)
        Coordinates of the top left corner.

    dim : int (default 2)
        Number of dimensions. When dim > 2, all extra dimensions are
        set to 0.

    Returns
    -------
    pos : dict
        A dictionary of positions keyed by node.

    Examples
    --------
    >>> G = dnx.chimera_graph(1)
    >>> pos = dnx.chimera_layout(G)

    """

    if not isinstance(G, nx.Graph):
        empty_graph = nx.Graph()
        empty_graph.add_nodes_from(G)
        G = empty_graph

    # best case scenario, each node in G has a chimera_index attribute. Otherwise
    # we will try to determine it using the find_chimera_indices function.
    if all('chimera_index' in dat for __, dat in G.nodes(data=True)):
        chimera_indices = {v: dat['chimera_index'] for v, dat in G.nodes(data=True)}
    else:
        chimera_indices = find_chimera_indices(G)

    # we could read these off of the name attribute for G, but we would want the values in
    # the nodes to override the name in case of conflict.
    m = max(idx[0] for idx in itervalues(chimera_indices)) + 1
    n = max(idx[1] for idx in itervalues(chimera_indices)) + 1
    t = max(idx[3] for idx in itervalues(chimera_indices)) + 1

    # ok, given the chimera indices, let's determine the coordinates
    xy_coords = chimera_node_placer_2d(m, n, t, scale, center, dim)
    pos = {v: xy_coords(i, j, u, k) for v, (i, j, u, k) in iteritems(chimera_indices)}

    return pos


def chimera_node_placer_2d(m, n, t, scale=1., center=None, dim=2):
    """Generates a function that converts chimera-indices to x, y
    coordinates for a plot.

    Parameters
    ----------
    m : int
        The number of rows in the Chimera lattice.

    n : int
        The number of columns in the Chimera lattice.

    t : int
        The size of the shore within each Chimera tile.

    scale : float (default 1.)
        Scale factor. When scale = 1 the all positions will fit within [0, 1]
        on the x-axis and [-1, 0] on the y-axis.

    center : None or array (default None)
        Coordinates of the top left corner.

    dim : int (default 2)
        Number of dimensions. When dim > 2, all extra dimensions are
        set to 0.

    Returns
    -------
    xy_coords : function
        A function that maps a Chimera-index (i, j, u, k) in an
        (m, n, t) Chimera lattice to x,y coordinates as could be
        used by a plot.

    """
    import numpy as np

    tile_center = t // 2
    tile_length = t + 3  # 1 for middle of cross, 2 for spacing between tiles
    scale /= max(m, n) * tile_length - 3  # want the enter plot to fill in [0, 1] when scale=1

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


def draw_chimera(G, linear_biases={}, quadratic_biases={},
                 nodelist=None, edgelist=None, cmap=None, edge_cmap=None, vmin=None, vmax=None,
                 edge_vmin=None, edge_vmax=None,
                 **kwargs):
    """Draw graph G with a Chimera layout.

    If linear_biases and/or quadratic_biases are provided then the biases
    are visualized on the plot.

    Parameters
    ----------
    G : graph
        A networkx graph. Should be a Chimera graph or a subgraph of a
        Chimera graph.

    linear_biases : dict (optional, default {})
        A dict of biases associated with each node in G. Should be of the
        form {node: bias, ...}. Each bias should be numeric.

    quadratic biases : dict (optional, default {})
        A dict of biases associated with each edge in G. Should be of the
        form {edge: bias, ...}. Each bias should be numeric. Self-loop
        edges are treated as linear biases.

    kwargs : optional keywords
       See networkx.draw_networkx() for a description of optional keywords,
       with the exception of the pos parameter which is not used by this
       function. If linear_biases or quadratic_biases are provided, then
       any provided node_color or edge_color arguments are ignored.

    """

    if linear_biases or quadratic_biases:
        # if linear biases and/or quadratic biases are provided, then color accordingly.

        try:
            import matplotlib.pyplot as plt
            import matplotlib as mpl
        except ImportError:
            raise ImportError("Matplotlib and numpy required for draw_chimera()")

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

    draw(G, chimera_layout(G), nodelist=nodelist, edgelist=edgelist,
         cmap=cmap, edge_cmap=edge_cmap, vmin=vmin, vmax=vmax, edge_vmin=edge_vmin,
         edge_vmax=edge_vmax,
         **kwargs)

    # if the biases are provided, then add a legend explaining the color map
    if linear_biases or quadratic_biases:
        fig = plt.figure(1)
        # cax = fig.add_axes([])
        cax = fig.add_axes([.9, 0.2, 0.04, 0.6])  # left, bottom, width, height
        mpl.colorbar.ColorbarBase(cax, cmap=cmap,
                                  norm=mpl.colors.Normalize(vmin=-1 * vmag, vmax=vmag, clip=False),
                                  orientation='vertical')
