# Copyright 2021 D-Wave Systems Inc.
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

from dwave_networkx import chimera_graph
from dwave_networkx.exceptions import DWaveNetworkXException
from dimod import DiscreteQuadraticModel
import networkx as nx
import numpy as np

__all__ = ["partition", "weighted_partition"]


def partition(G, num_partitions=2, lagrange=4, sampler=None, **sampler_args):
    """Returns an approximate k-partition of G.

    Defines an DQM with ground states corresponding to a
    balanced k-partition of G and uses the sampler to sample from it.

    A k-partition is a collection of k subsets of the vertices
    of G such that each vertex is in exactly one subset, and
    the number of edges between vertices in different subsets
    is as small as possible.

    Parameters
    ----------
    G : NetworkX graph
        The graph to partition.

    num_partitions : int, optional (default 2)
        The number of subsets in the desired partition.

    lagrange: float, optional (default 4)
        The value of the lagrange parameter used to enforce
        the constraint that each partition is of equal size.

    sampler
        A discrete quadratic model sampler. A sampler is a process that
        samples from low energy states in models defined by an Ising
        equation or a Quadratic Unconstrained Binary Optimization
        Problem (QUBO). The sampler is expected to have a 'sample_dqm'
        method. A sampler is expected to return an iterable of samples,
        in order of increasing energy.

    sampler_args
        Additional keyword parameters are passed to the sampler.

    Returns
    -------
    node_partition : dict
        The partition as a dictionary mapping each node to subsets labelled
        as integers {0, 1, 2, ... num_partitions}.

    Example
    -------
    This example uses a sampler from
    `dimod <https://github.com/dwavesystems/dimod>`_ to find a 2-partition
    for a graph of a Chimera unit cell created using the `chimera_graph()`
    function.

    >>> import dimod
    >>> from dwave.system import LeapHybridDQMSampler
    ...
    >>> sampler = LeapHybridDQMSampler()
    >>> G = dnx.chimera_graph(1, 1, 4)
    >>> partitions = dnx.partition(G, sampler=sampler)

    Notes
    -----
    Samplers by their nature may not return the optimal solution. This
    function does not attempt to confirm the quality of the returned
    sample.

    """

    dqm = graph_partition_dqm(G, num_partitions, lagrange, weighted=False)

    # Solve the problem using the DQM solver
    node_partition = sampler.sample_dqm(dqm, **sampler_args).first.sample

    return node_partition


def weighted_partition(G, num_partitions=2, lagrange=4, sampler=None, **sampler_args):
    """Returns an approximate weighted k-partition.

    Defines an Ising problem with ground states corresponding to
    a weighted k-partition and uses the sampler to sample from it.

    A weighted k-partition is a collection of k subsets of the vertices
    of G such that each vertex is in exactly one subset, and
    the sum of the total weights of edges between vertices in different
    subsets is as small as possible.

    Parameters
    ----------
    G : NetworkX graph
        The graph on which to find a weighted k-partition. Each edge in G should
        have a numeric `weight` attribute.

    num_partitions : int, optional (default 2)
        The number of subsets in the desired partition.

    lagrange: float, optional (default 4)
        The value of the lagrange parameter used to enforce
        the constraint that each partition is of equal size.
        
    sampler
        A discrete quadratic model sampler. A sampler is a process that
        samples from low energy states in models defined by an Ising
        equation or a Quadratic Unconstrained Binary Optimization
        Problem (QUBO). The sampler is expected to have a 'sample_dqm'
        method. A sampler is expected to return an iterable of samples,
        in order of increasing energy.

    sampler_args
        Additional keyword parameters are passed to the sampler.

    Returns
    -------
    node_partition : dict
        The partition as a dictionary mapping each node to subsets labelled
        as integers from 0 to num_partitions.

    Notes
    -----
    Samplers by their nature may not return the optimal solution. This
    function does not attempt to confirm the quality of the returned
    sample.

    """
    dqm = graph_partition_dqm(G, num_partitions, lagrange, weighted=True)

    # Solve the problem using the DQM solver
    node_partition = sampler.sample_dqm(dqm, **sampler_args).first.sample

    return node_partition


def graph_partition_dqm(G, num_partitions, lagrange, weighted=False):
    # Set up partitions
    partitions = range(num_partitions)

    # Initialize the DQM object
    dqm = DiscreteQuadraticModel()

    # Define the DQM variables. We need to define all of them first because there
    # are not edges between all the nodes; hence, there may be quadratic terms
    # between nodes which don't have edges connecting them.
    for p in G.nodes:
        dqm.add_variable(num_partitions, label=p)

    constraint_const = lagrange * (1 - (2 * G.number_of_nodes() / num_partitions))
    for p in G.nodes:
        linear_term = constraint_const + (0.5 * np.ones(num_partitions) * G.degree(p, weight='weight'))
        dqm.set_linear(p, linear_term)

    # Quadratic term for node pairs which do not have edges between them
    for p0, p1 in nx.non_edges(G):
        dqm.set_quadratic(p0, p1, {(c, c): (2 * lagrange) for c in partitions})

    # Quadratic term for node pairs which have edges between them
    if weighted:
        try:
            for p0, p1 in G.edges:
                dqm.set_quadratic(p0, p1, {(c, c): ((2 * lagrange) - G[p0][p1]['weight']) for c in partitions})
        except KeyError:
            raise DWaveNetworkXException("edges must have 'weight' attribute")
    else:
        for p0, p1 in G.edges:
            dqm.set_quadratic(p0, p1, {(c, c): ((2 * lagrange) - 1) for c in partitions})

    return dqm