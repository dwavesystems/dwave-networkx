.. image:: https://img.shields.io/pypi/v/dwave-networkx.svg
    :target: https://pypi.org/project/dwave-networkx

.. image:: https://codecov.io/gh/dwavesystems/dwave-networkx/branch/main/graph/badge.svg
    :target: https://codecov.io/gh/dwavesystems/dwave-networkx

.. image:: https://circleci.com/gh/dwavesystems/dwave-networkx.svg?style=svg
    :target: https://circleci.com/gh/dwavesystems/dwave-networkx

.. inclusion-marker-do-not-remove

D-Wave NetworkX
===============

.. index-start-marker

D-Wave NetworkX is an extension of `NetworkX <https://networkx.org>`_\ ---a
Python language package for exploration and analysis of networks and network
algorithms---for users of D-Wave quantum computers. It provides tools for working 
with Quantum Processing Unit (QPU) topology graphs, such as the Pegasus used on 
the Advantage\ |TM| system, and implementations of graph-theory algorithms on D-Wave
quantum computers and other binary quadratic model samplers.

.. |TM| replace:: :sup:`TM`

This example generates a Pegasus graph of the size used by Advantage QPUs.

>>> import dwave_networkx as dnx
>>> graph = dnx.pegasus_graph(16)

See the documentation for more examples.

.. index-end-marker

Installation
============

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
=======

Released under the Apache License 2.0.

Contributing
============

Ocean's `contributing guide <https://docs.ocean.dwavesys.com/en/stable/contributing.html>`_
has guidelines for contributing to Ocean packages.
