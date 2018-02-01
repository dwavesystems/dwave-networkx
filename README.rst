
.. image:: https://travis-ci.org/dwavesystems/dwave_networkx.svg?branch=master
    :target: https://travis-ci.org/dwave_networkx

.. inclusion-marker-do-not-remove

D-Wave NetworkX
====================

An extension of `NetworkX <http://networkx.github.io>`_\ ---a Python language package for exploration
and analysis of networks and network algorithms---D-Wave NetworkX extension has three primary goals:

* Include graphs and algorithms relevant to working with the D-Wave System.
* Allow for easy visualization of Chimera-structured graphs.
* Provide an implementation of some graph theory algorithms that uses the D-Wave System or another binary quadratic model sampler.

Example Usage
----------------

.. code: python

>>> import dwave_networkx as dnx
>>> graph = dnx.chimera_graph(1, 1, 4)
>>> list(graph.nodes())
[0, 1, 2, 3, 4, 5, 6, 7]
>>> list(graph.edges())
[(0, 4), (0, 5), (0, 6), (0, 7), (1, 4), (1, 5), (1, 6), (1, 7), (2, 4), (2, 5), (2, 6), (2, 7), (3, 4), (3, 5), (3, 6), (3, 7)]

See :ref:`reference` for more examples.

Installation
====================

.. installation-start-marker

**Installation from PyPi:**

.. code-block:: bash

  pip install dwave_networkx

**Installation from source:**

.. code-block:: bash

  python setup.py install

.. installation-end-marker

License
====================

Released under the Apache License 2.0.

.. include:: ../LICENSE.txt
