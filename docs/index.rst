.. _index_dnx:

==============
dwave-networkx
==============

.. dropdown:: About dwave-networkx

    .. include:: README.rst
        :start-after: start_dnx_about
        :end-before: end_dnx_about

    D-Wave NetworkX provides tools for working with Quantum Processing Unit (QPU) 
    topology graphs, such as the :term:`Pegasus` used on the Advantage\ |TM| system,
    and implementations of graph-theory algorithms on D-Wave quantum computers and 
    other binary quadratic model :term:`sampler`\ s; for example, functions such as
    :func:`.draw_pegasus` provide easy visualization for Pegasus graphs; functions 
    such as :func:`~dwave_networkx.algorithms.max_cut.maximum_cut` or 
    :func:`~dwave_networkx.algorithms.cover.min_vertex_cover` provide graph algorithms 
    useful to optimization problems that fit well with D-Wave quantum computers.

    Like D-Wave quantum computers, all other supported samplers must have
    ``sample_qubo`` and ``sample_ising`` methods for solving :term:`Ising` and 
    :term:`QUBO` models and return an iterable of samples in order of increasing 
    energy. You can set a default sampler using the 
    :func:`~dwave_networkx.default_sampler.set_default_sampler` function.

    *   For an introduction to quantum processing unit (QPU) topologies such as the
        Pegasus graph, see :std:doc:`Topology <oceandocs:concepts/topology>`.
    *   For an introduction to binary quadratic models (BQMs), see
        :std:doc:`Binary Quadratic Models <oceandocs:concepts/bqm>`.
    *   For an introduction to samplers, see
        :std:doc:`Samplers and Composites <oceandocs:concepts/samplers>`.

    This example creates a Pegasus graph (used by Advantage) and a small Zephyr graph 
    (used by the Advantage2\ |TM| prototype made available in Leap\ |TM| in June 2022):

    .. |TM| replace:: :sup:`TM`

    >>> import dwave_networkx as dnx
    ...
    >>> # Advantage
    >>> P16 = dnx.pegasus_graph(16)
    ...
    >>> # Advantage2
    >>> Z4 = dnx.zephyr_graph(4)

Reference Documentation
=======================

.. toctree::
    :maxdepth: 1

    reference/api_ref