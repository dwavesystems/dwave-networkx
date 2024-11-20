********
Coloring
********

Graph coloring is the problem of assigning a color to the vertices of a graph
in a way that no adjacent vertices have the same color.

Example
-------

The map-coloring problem is to assign a color to each region of a map
(represented by a vertex on a graph) such that any two regions sharing a
border (represented by an edge of the graph) have different colors.

.. figure:: ../../_images/Problem_MapColoring.png
   :name: Problem_MapColoringColoring
   :alt: image
   :align: center
   :scale: 70 %

   Coloring a map of Canada with four colors.

.. automodule:: dwave_networkx.algorithms.coloring
.. autosummary::
   :toctree: generated/

   is_vertex_coloring
   min_vertex_color
   min_vertex_color_qubo
   vertex_color
   vertex_color_qubo
