*********
Utilities
*********

.. automodule:: dwave_networkx.utils
.. currentmodule:: dwave_networkx.utils

Decorators
----------
.. automodule:: dwave_networkx.utils.decorators
.. autosummary::
   :toctree: generated/

   binary_quadratic_model_sampler

.. currentmodule:: dwave_networkx

.. toctree::
   :hidden:

   utilities/index

Graph Indexing
--------------

See :ref:`index_conversion_classes` on instantiating the needed lattice size
and setting correct domain and range for coordinates in a QPU working graph.

For the iterator versions of these functions, see the code.

Chimera
~~~~~~~

.. autosummary::
   :toctree: generated/

   chimera_coordinates.chimera_to_linear
   chimera_coordinates.linear_to_chimera
   find_chimera_indices

Pegasus
~~~~~~~

.. autosummary::
   :toctree: generated/

   pegasus_coordinates.linear_to_nice
   pegasus_coordinates.linear_to_pegasus
   pegasus_coordinates.nice_to_linear
   pegasus_coordinates.nice_to_pegasus
   pegasus_coordinates.pegasus_to_linear
   pegasus_coordinates.pegasus_to_nice


Zephyr
~~~~~~

.. autosummary::
   :toctree: generated/

   zephyr_coordinates.graph_to_linear
   zephyr_coordinates.graph_to_zephyr
   zephyr_coordinates.iter_linear_to_zephyr
   zephyr_coordinates.iter_linear_to_zephyr_pairs
   zephyr_coordinates.iter_zephyr_to_linear
   zephyr_coordinates.iter_zephyr_to_linear_pairs
   zephyr_coordinates.linear_to_zephyr
   zephyr_coordinates.zephyr_to_linear
   zephyr_sublattice_mappings


Exceptions
----------
.. automodule:: dwave_networkx.exceptions
.. autosummary::
   :toctree: generated/

   DWaveNetworkXException
   DWaveNetworkXMissingSampler
