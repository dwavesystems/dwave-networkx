..  -*- coding: utf-8 -*-

.. _contents:

Overview
========

NetworkX
--------

NetworkX is a Python language package for exploration and analysis of networks and network algorithms [NX]_.

NetworkX can be found at http://networkx.github.io.


D-Wave NetworkX
---------------

The D-Wave NetworkX extension has three primary goals:

* Include graphs and algorithms relevant to working with the D-Wave System.
* Allow for easy visualization of Chimera-structured graphs.
* Provide an implementation of some graph theory algorithms that uses the D-Wave System or another binary quadratic model sampler.

Information about the D-Wave system can be found at https://www.dwavesys.com/.

Relationship to dimod
---------------------

D-Wave NetworkX's algorithms rely on binary quadratic model samplers. Binary quadratic models are the model class that contains Quadratic Unconstrained Binary Optimization Problems (QUBOs_) and Ising_ models. A sampler is a process that samples from low energy states in models defined by an Ising equation or a QUBO's quadratic polynomial.

To work with D-Wave NetworkX, a sampler object is expected to have a ‘sample_qubo’ and ‘sample_ising’ method. A sampler is expected to return an iterable of samples, in order of increasing energy.

dimod_ provides a shared API for samplers that fulfills the above as well as a few simple samplers that can be used to get started quickly

.. _dimod: https://github.com/dwavesystems/dimod
.. _Ising: https://en.wikipedia.org/wiki/Ising_model
.. _QUBOs: https://en.wikipedia.org/wiki/Quadratic_unconstrained_binary_optimization

Documentation
-------------

.. only:: html

    :Release: |version|
    :Date: |today|

.. toctree::
   :maxdepth: 1

   reference/index
   about_dwave
   license
   bibliography

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
* :ref:`glossary`