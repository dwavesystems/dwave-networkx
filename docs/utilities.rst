.. _graphs_utilities:

=========
Utilities
=========

.. currentmodule:: dwave.graphs

Graph Indexing
==============

See the :ref:`graphs_coordinates_conversion` subsection on instantiating the needed
lattice size and setting correct domain and range for coordinates in a QPU
working graph. For the iterator versions of these functions, see the code.

Chimera
-------

.. autosummary::
   :toctree: generated/

   chimera_coordinates.chimera_to_linear
   chimera_coordinates.graph_to_chimera
   chimera_coordinates.graph_to_linear
   chimera_coordinates.linear_to_chimera
   chimera_sublattice_mappings
   find_chimera_indices

Pegasus
-------

.. autosummary::
   :toctree: generated/

   pegasus_coordinates.graph_to_linear
   pegasus_coordinates.graph_to_nice
   pegasus_coordinates.graph_to_pegasus
   pegasus_coordinates.linear_to_nice
   pegasus_coordinates.linear_to_pegasus
   pegasus_coordinates.nice_to_linear
   pegasus_coordinates.nice_to_pegasus
   pegasus_coordinates.pegasus_to_linear
   pegasus_coordinates.pegasus_to_nice
   pegasus_sublattice_mappings


Zephyr
------

.. autosummary::
   :toctree: generated/

   zephyr_coordinates.graph_to_linear
   zephyr_coordinates.graph_to_zephyr
   zephyr_coordinates.linear_to_zephyr
   zephyr_coordinates.zephyr_to_linear
   zephyr_sublattice_mappings


.. _graphs_coordinates_conversion:

Coordinates Conversion
----------------------

.. automodule:: dwave.graphs

.. autoclass:: chimera_coordinates

.. autoclass:: pegasus_coordinates

.. autoclass:: zephyr_coordinates



Exceptions
==========

.. automodule:: dwave.graphs.exceptions
.. autosummary::
   :toctree: generated/

   DWaveGraphsError
