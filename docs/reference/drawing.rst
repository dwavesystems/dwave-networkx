.. _drawing:

Drawing
*******

.. currentmodule:: dwave_networkx

.. automodule:: dwave_networkx.drawing.chimera_layout

.. note:: Some functionality requires `NumPy <http://scipy.org>`_ and/or
	 `Matplotlib <https://matplotlib.org>`_\ .

Chimera Graph Functions
-----------------------

.. autosummary::
   :toctree: generated/

   chimera_layout
   draw_chimera

Example
---------

This example uses the `chimera_layout()` function to show the positions of nodes of a simple
5-node NetworkX graph in a Chimera lattice. It then uses the `chimera_graph()`
and `draw_chimera()` functions to display those positions on a Chimera unit cell.

.. code-block:: python

   >>> import networkx as nx
   >>> import dwave_networkx as dnx
   >>> import matplotlib.pyplot as plt
   >>> H = nx.Graph()
   >>> H.add_nodes_from([0, 4, 5, 6, 7])
   >>> H.add_edges_from([(0, 4), (0, 5), (0, 6), (0, 7)])
   >>> pos=dnx.chimera_layout(H)
   >>> pos
   {0: array([ 0. , -0.5]),
    4: array([ 0.5,  0. ]),
    5: array([ 0.5 , -0.25]),
    6: array([ 0.5 , -0.75]),
    7: array([ 0.5, -1. ])}
   >>> # Show graph H on a Chimera unit cell
   >>> plt.ion()
   >>> G=dnx.chimera_graph(1, 1, 4)  # Draw a Chimera unit cell
   >>> dnx.draw_chimera(G)
   >>> dnx.draw_chimera(H, node_color='b', node_shape='*', style='dashed', edge_color='b', width=3)
   >>> # matplotlib commands to add labels to graphic not shown



.. figure:: ../_images/chimera_layout_0-rightside.png
	:align: center
	:name: chimera_layout_0-rightside
	:scale: 60 %
	:alt: Graph H overlaid on a Chimera unit cell.

	Graph H (blue) overlaid on a Chimera unit cell (red nodes and black edges).
