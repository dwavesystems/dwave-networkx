
.. image:: https://travis-ci.org/dwavesystems/dwave_networkx.svg?branch=master
    :target: https://travis-ci.org/dwavesystems/dwave_networkx

.. inclusion-marker-do-not-remove

D-Wave NetworkX
====================

D-Wave NetworkX is an extension of `NetworkX <http://networkx.github.io>`_\ ---a
Python language package for exploration and analysis of networks and network
algorithms---for users of D-Wave Systems. It provides tools for working with
Chimera graphs and implementations of graph-theory algorithms on the D-Wave
system and other binary quadratic model samplers.

Example Usage
----------------

This example generates a graph for a Chimera unit cell (eight nodes in a 4-by-2
bipartite architecture). 

.. code: python

>>> import dwave_networkx as dnx
>>> graph = dnx.chimera_graph(1, 1, 4)
>>> list(graph.nodes())
[0, 1, 2, 3, 4, 5, 6, 7]
>>> list(graph.edges())
[(0, 4), (0, 5), (0, 6), (0, 7), (1, 4), (1, 5), (1, 6), (1, 7), (2, 4), (2, 5), (2, 6), (2, 7), (3, 4), (3, 5), (3, 6), (3, 7)]

See the documentation for more examples.

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
