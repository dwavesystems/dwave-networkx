***********
Maximum Cut
***********

A maximum cut is a subset of a graph's vertices such that the number of edges
between this subset and the remaining vertices is as large as possible.

.. figure:: ../../_images/MaxCut.png
   :name: Cut
   :alt: image
   :align: center
   :scale: 60 %

   Maximum cut for a Chimera unit cell: the blue line around the subset of nodes
   {4, 5, 6, 7} cuts 16 edges; adding or removing a node decreases
   the number of edges between the two complementary subsets of the graph.

.. automodule:: dwave_networkx.algorithms.max_cut
.. autosummary::
   :toctree: generated/

    maximum_cut
    weighted_maximum_cut
