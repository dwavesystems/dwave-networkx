.. image:: https://img.shields.io/pypi/v/dwave-graphs.svg
    :target: https://pypi.org/project/dwave-graphs

.. image:: https://codecov.io/gh/dwavesystems/dwave-graphs/branch/main/graph/badge.svg
    :target: https://codecov.io/gh/dwavesystems/dwave-graphs

.. image:: https://circleci.com/gh/dwavesystems/dwave-networkx.svg?style=svg
    :target: https://circleci.com/gh/dwavesystems/dwave-graphs

============
dwave-graphs
============

.. start_graphs_about

dwave-graphs provides tools for
working with quantum processing unit (QPU) topology graphs, such as the Pegasus
used on the Advantage\ :sup:`TM` quantum computer, and implementations of
graph-theory algorithms on D-Wave quantum computers and other binary quadratic
model (BQM) samplers.

This example generates a Pegasus graph of the size used by Advantage QPUs.

>>> import dwave.graphs
>>> graph = dwave.graphs.pegasus_graph(16)

.. end_graphs_about

Installation
============

**Installation from PyPi:**

.. code-block:: bash

    pip install dwave-graphs

**Installation from source:**

.. code-block:: bash

    pip install -r requirements.txt
    python setup.py install

License
=======

Released under the Apache License 2.0.

Contributing
============

Ocean's `contributing guide <https://docs.dwavequantum.com/en/latest/ocean/contribute.html>`_
has guidelines for contributing to Ocean packages.
