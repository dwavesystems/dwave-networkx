***************
Independent Set
***************

An independent set is a set of a graph's vertices with no edge connecting any
of its member pairs.

.. figure:: ../../_images/Cover.png
   :name: CoverIndependentSet
   :alt: image
   :align: center
   :scale: 40 %

   Independent sets for a Chimera unit cell: the nodes of both the blue set
   of vertices (the horizontal tile of the Chimera unit cell) and the red set
   (vertical tile) are independent sets of the graph, with no blue node adjacent
   to another blue node and likewise for red nodes.

.. automodule:: dwave_networkx.algorithms.independent_set

.. currentmodule:: dwave_networkx

.. autosummary::
   :toctree: generated/

    maximum_weighted_independent_set
    maximum_independent_set
    is_independent_set

Helper Functions
----------------

.. currentmodule:: dwave_networkx.algorithms.independent_set

.. autosummary::
   :toctree: generated/

   maximum_weighted_independent_set_qubo

