.. image:: https://img.shields.io/pypi/v/dwave-networkx.svg
    :target: https://pypi.python.org/pypi/dwave-networkx

.. image:: https://readthedocs.com/projects/d-wave-systems-dwave-networkx/badge/?version=latest
    :target: https://docs.ocean.dwavesys.com/projects/dwave-networkx/en/latest/?badge=latest

.. image:: https://codecov.io/gh/dwavesystems/dwave_networkx/branch/master/graph/badge.svg
    :target: https://codecov.io/gh/dwavesystems/dwave_networkx

.. image:: https://circleci.com/gh/dwavesystems/dwave_networkx.svg?style=svg
    :target: https://circleci.com/gh/dwavesystems/dwave_networkx

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

See the documentation for more examples.

Installation
====================

.. installation-start-marker

**Installation from PyPi:**

.. code-block:: bash

  pip install dwave_networkx

**Installation from source:**

.. code-block:: bash

  pip install -r requirements.txt
  python setup.py install

.. installation-end-marker

License
====================

Released under the Apache License 2.0.
