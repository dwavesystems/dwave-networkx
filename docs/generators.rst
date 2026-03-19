.. _graphs_generators:

================
Graph Generators
================

Generators for graphs, such the graphs (topologies) of D-Wave System QPUs.

.. currentmodule:: dwave.graphs

D-Wave Systems
==============

.. autosummary::
   :toctree: generated/

   chimera_graph
   pegasus_graph
   zephyr_graph

Example
-------

This example uses the the `chimera_graph()` function to create a Chimera lattice
of size (1, 1, 4), which is a single unit cell in Chimera topology, and
the `find_chimera()` function to determine the Chimera indices.

.. code-block:: python

  >>> import networkx as nx
  >>> import dwave.graphs
  >>> G = dwave.graphs.chimera_graph(1, 1, 4)
  >>> chimera_indices = dwave.graphs.find_chimera_indices(G)
  >>> print chimera_indices
  {0: (0, 0, 0, 0),
   1: (0, 0, 0, 1),
   2: (0, 0, 0, 2),
   3: (0, 0, 0, 3),
   4: (0, 0, 1, 0),
   5: (0, 0, 1, 1),
   6: (0, 0, 1, 2),
   7: (0, 0, 1, 3)}

.. figure:: _images/find_chimera-unitcell.png
	:align: center
	:name: find_chimera-unitcell
	:scale: 30 %
	:alt: Indices of a Chimera unit cell.

	Indices of a Chimera unit cell found by creating a lattice of size (1, 1, 4).

Toruses
=======

.. autosummary::
   :toctree: generated/

   chimera_torus
   pegasus_torus
   zephyr_torus

Other Graphs
============

.. autosummary::
   :toctree: generated/

   markov_network
