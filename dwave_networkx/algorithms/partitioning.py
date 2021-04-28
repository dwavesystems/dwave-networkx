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

from dwave_networkx.exceptions import DWaveNetworkXException
from dwave_networkx.utils import binary_quadratic_model_sampler
from dwave_networkx import chimera_graph
from dimod import DiscreteQuadraticModel
from dwave.system import LeapHybridDQMSampler
import networkx as nx
import numpy as np

__all__ = ["partition", "weighted_partition"]


#@binary_quadratic_model_sampler(1)
def partition(G, num_partitions=2, lagrange=4, sampler=None, **sampler_args):
    """Returns an approximate k-partition of G.

    Defines an Ising problem with ground states corresponding to a
    balanced k-partition of G and uses the sampler to sample from it.

    A k-partition is a collection of k subsets of the vertices
    of G such that each vertex is in exactly one subset, and
    the sum of the number of edges between vertices in different
    subsets is as small as possible.

    Parameters
    ----------
    G : NetworkX graph
        The graph on which to find a partition.

    num_partitions : int, optional (default 2)
        The number of subsets in the desired partition.

    lagrange: float, optional (default 4)
        The value of the lagrange parameter used to enforce
        the constraint that each partition is of equal size.

    sampler
        A discrete quadratic model sampler. A sampler is a process that
        samples from low energy states in models defined by an Ising
        equation or a Quadratic Unconstrained Binary Optimization
        Problem (QUBO). A sampler is expected to have a 'sample_dqm'
        method. A sampler is expected to return an iterable of samples,
        in order of increasing energy. If no sampler is provided, one
        must be provided using the `set_default_sampler` function.

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
    ...
    >>> sampler = LeapHybridDQMSampler()
    >>> G = dnx.chimera_graph(1, 1, 4)
    >>> partitions = dnx.partition(G, sampler)

    Notes
    -----
    Samplers by their nature may not return the optimal solution. This
    function does not attempt to confirm the quality of the returned
    sample.

    """

    # Set up partitions
    partitions = range(num_partitions)
    num_nodes = G.number_of_nodes()

    # Initialize the DQM object
    dqm = DiscreteQuadraticModel()

    # Define the DQM variables. We need to define all of them first because there
    # are not edges between all the nodes; hence, there may be quadratic terms
    # between nodes which don't have edges connecting them.
    for p in G.nodes:
        dqm.add_variable(num_partitions, label=p)

    constraint_const = lagrange * (1 - (2 * num_nodes / num_partitions))
    for p in G.nodes:
        linear_term = constraint_const + (0.5 * np.ones(num_partitions) * G.degree[p])
        dqm.set_linear(p, linear_term)

    # Quadratic term for node pairs which do not have edges between them
    for p0, p1 in nx.non_edges(G):
        dqm.set_quadratic(p0, p1, {(c, c): (2 * lagrange) for c in partitions})

    # Quadratic term for node pairs which have edges between them
    for p0, p1 in G.edges:
        dqm.set_quadratic(p0, p1, {(c, c): ((2 * lagrange) - 1) for c in partitions})

    # Solve the problem using the DQM solver
    offset = lagrange * num_nodes * num_nodes / num_partitions
    sampleset = sampler.sample_dqm(dqm)#, label='Graph Partitioning DQM')

    node_partition = sampleset.first.sample

    return node_partition

def weighted_partition(G, num_partitions=2, lagrange=4, sampler=None, **sampler_args):
    """Returns an approximate weighted maximum cut.

    Defines an Ising problem with ground states corresponding to
    a weighted k-partition and uses the sampler to sample from it.

    A weighted k-partition is a collection of k subsets of the vertices
    of G such that each vertex is in exactly one subset, and
    the sum of the total weights of edges between vertices in different
    subsets is as small as possible.

    Parameters
    ----------
    G : NetworkX graph
        The graph on which to find a weighted maximum cut. Each edge in G should
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
        Problem (QUBO). A sampler is expected to have a 'sample_qubo'
        and 'sample_ising' method. A sampler is expected to return an
        iterable of samples, in order of increasing energy. If no
        sampler is provided, one must be provided using the
        `set_default_sampler` function.

    sampler_args
        Additional keyword parameters are passed to the sampler.

    Returns
    -------
    node_partition : dict
        The partition as a dictionary mapping each node to subsets labelled
        as integers {0, 1, 2, ... num_partitions}.

    Notes
    -----
    Samplers by their nature may not return the optimal solution. This
    function does not attempt to confirm the quality of the returned
    sample.

    """
    # Set up partitions
    partitions = range(num_partitions)
    num_nodes = G.number_of_nodes()

    # Initialize the DQM object
    dqm = DiscreteQuadraticModel()

    # Define the DQM variables. We need to define all of them first because there
    # are not edges between all the nodes; hence, there may be quadratic terms
    # between nodes which don't have edges connecting them.
    for p in G.nodes:
        dqm.add_variable(num_partitions, label=p)

    constraint_const = lagrange * (1 - (2 * num_nodes / num_partitions))
    for p in G.nodes:
        linear_term = constraint_const + (0.5 * np.ones(num_partitions) * G.degree[p])
        dqm.set_linear(p, linear_term)

    # Quadratic term for node pairs which do not have edges between them
    for p0, p1 in nx.non_edges(G):
        dqm.set_quadratic(p0, p1, {(c, c): (2 * lagrange) for c in partitions})

    # Quadratic term for node pairs which have edges between them
    for p0, p1 in G.edges:
        dqm.set_quadratic(p0, p1, {(c, c): ((2 * lagrange) - G[p0][p1]["weight"]) for c in partitions})

    # Solve the problem using the DQM solver
    offset = lagrange * num_nodes * num_nodes / num_partitions
    sampleset = sampler.sample_dqm(dqm)#, label='Graph Partitioning DQM')

    node_partition = sampleset.first.sample

    return node_partition