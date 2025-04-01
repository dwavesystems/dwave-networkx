.. _index_dnx:

==============
dwave-networkx
==============


.. toctree::
    :caption: Reference documentation for dwave-networkx:
    :maxdepth: 1

    api_ref


About dwave-networkx
====================

.. include:: README.rst
    :start-after: start_dnx_about
    :end-before: end_dnx_about

Functions such as :func:`.draw_pegasus` provide easy visualization for
:term:`Pegasus` graphs while functions such as
:func:`~dwave_networkx.algorithms.max_cut.maximum_cut` or
:func:`~dwave_networkx.algorithms.cover.min_vertex_cover` provide graph
algorithms useful to optimization problems that fit well with |dwave_short|
quantum computers.

Like |dwave_short| quantum computers, all other supported samplers must have
``sample_qubo`` and ``sample_ising`` methods for solving :term:`Ising` and
:term:`QUBO` models and return an iterable of samples in order of increasing
energy. You can set a default sampler using the
:func:`~dwave_networkx.default_sampler.set_default_sampler` function.

Usage Information
=================

*   :ref:`index_concepts` for terminology
*   :ref:`qpu_topologies` for an introduction to :term:`QPU`
    :term:`topologies <topology>` such as the :term:`Pegasus` graph
*   :ref:`concept_models_bqm` for an introduction to binary quadratic models
    (:term:`BQM`)
*   :ref:`concept_samplers` for an introduction to :term:`samplers <sampler>`