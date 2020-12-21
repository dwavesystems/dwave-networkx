********
Matching
********

A matching is a subset of graph edges in which no vertex occurs more than once.

.. figure:: ../../_images/Match.png
   :name: Matching
   :alt: image
   :align: center
   :scale: 40 %

   A matching for a Chimera unit cell: no vertex is incident to more than one
   edge in the set of blue edges

.. automodule:: dwave.plugins.networkx.algorithms.matching
.. autosummary::
   :toctree: generated/

    matching_bqm
    maximal_matching_bqm
    min_maximal_matching_bqm
    min_maximal_matching
