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

.. figure:: ../../_static/Problem_MapColoring.png
   :name: Problem_MapColoring
   :alt: image
   :align: center
   :scale: 70 %

   Coloring a map of Canada with four colors.

.. automodule:: dwave_networkx.algorithms.coloring
.. autosummary::
   :toctree: generated/

   min_vertex_coloring
   is_vertex_coloring
