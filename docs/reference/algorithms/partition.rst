*************
Partitioning
*************

A `k`-partition consists of `k` disjoint and equally sized subsets of a
graph's vertices such that the total number of edges between nodes in
distinct subsets is as small as possible.

.. figure:: ../../_images/Partitioning.png
   :name: Partition
   :alt: image
   :align: center
   :scale: 60 %

   A 2-partition for a simple graph: the nodes in blue are in the
   '0' subset, and the nodes in red are in the '1' subset. There are no
   other arrangements with fewer edges between two equally sized subsets.

.. automodule:: dwave_networkx.algorithms.partition
.. autosummary::
   :toctree: generated/

    partition