*************
Partitioning
*************

A `k`-partition consists of `k` disjoint and equally sized subsets of a
graph's vertices such that the total number of edges between nodes in
distinct subsets is as small as possible.

.. figure:: ../../_images/Partition.png
   :name: Cut
   :alt: image
   :align: center
   :scale: 60 %

   A 2-partition for a Chimera unit cell: the nodes in blue are in the
   '0' subset, and the nodes in red are in the '1' subset. There are no
   other arrangements with few edges between two equally sized subsets.

.. automodule:: dwave_networkx.algorithms.partition
.. autosummary::
   :toctree: generated/

    partition
    weighted_partition
