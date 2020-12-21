.. image:: https://img.shields.io/pypi/v/dwave-networkx.svg
    :target: https://pypi.python.org/pypi/dwave-networkx

.. image:: https://readthedocs.com/projects/d-wave-systems-dwave-networkx/badge/?version=latest
    :target: https://docs.ocean.dwavesys.com/projects/dwave-networkx/en/latest/?badge=latest

.. image:: https://codecov.io/gh/dwavesystems/dwave-networkx/branch/main/graph/badge.svg
    :target: https://codecov.io/gh/dwavesystems/dwave-networkx

.. image:: https://circleci.com/gh/dwavesystems/dwave-networkx.svg?style=svg
    :target: https://circleci.com/gh/dwavesystems/dwave-networkx

.. inclusion-marker-do-not-remove

D-Wave NetworkX
===============

.. index-start-marker

D-Wave NetworkX is an extension of `NetworkX <http://networkx.github.io>`_\ ---a
Python language package for exploration and analysis of networks and network
algorithms---for users of D-Wave Systems. It provides tools for working with
Chimera graphs and implementations of graph-theory algorithms on the D-Wave
system and other binary quadratic model samplers.

The example below generates a graph for a Chimera unit cell (eight nodes in a 4-by-2
bipartite architecture).

.. code: python

>>> import dwave.plugins.networkx as dnx
>>> graph = dnx.chimera_graph(1, 1, 4)

See the documentation for more examples.

.. index-end-marker

Installation
============

.. installation-start-marker

**Installation from PyPi:**

.. code-block:: bash

  pip install dwave-networkx

**Installation from source:**

.. code-block:: bash

  pip install -r requirements.txt
  python setup.py install

.. installation-end-marker

License
=======

Released under the Apache License 2.0.
