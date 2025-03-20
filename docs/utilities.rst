.. _dnx_utilities:

=========
Utilities
=========

.. automodule:: dwave_networkx.utils
.. currentmodule:: dwave_networkx.utils

Decorators
==========

.. automodule:: dwave_networkx.utils.decorators
.. autosummary::
   :toctree: generated/

   binary_quadratic_model_sampler

.. currentmodule:: dwave_networkx

Graph Indexing
==============

See the :ref:`dnx_coordinates_conversion` subsection on instantiating the needed
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


.. _dnx_coordinates_conversion:

Coordinates Conversion
----------------------

.. automodule:: dwave_networkx

.. autoclass:: chimera_coordinates

.. autoclass:: pegasus_coordinates

.. autoclass:: zephyr_coordinates



Exceptions
==========

.. automodule:: dwave_networkx.exceptions
.. autosummary::
   :toctree: generated/

   DWaveNetworkXException
   DWaveNetworkXMissingSampler
