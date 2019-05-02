# Copyright 2018 D-Wave Systems Inc.
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
#
# ================================================================================================
"""
Generators for graphs derived from the D-Wave System.

"""
import networkx as nx

from dwave_networkx import _PY2
from dwave_networkx.exceptions import DWaveNetworkXException

from dimod.decorators import graph_argument

__all__ = ['shared_working_graph']

# compatibility for python 2/3
if _PY2:
    range = xrange


import numpy as np

@graph_argument('g1', 'g2')
def shared_working_graph(g1, g2):
    """Creates a graph using the common nodes and edges of two given Chimera or
    Pegasus graphs.

    In a D-Wave system, the set of qubits and couplers that are available for
    computation is known as the working graph.

    Parameters
    ----------
    g1: (int/tuple[nodes, edges]/:obj:`~networkx.Graph`)
        A Chimera or Pegasus graph. Either a nodes/edges pair
        or a NetworkX graph.

    g2: (int/tuple[nodes, edges]/:obj:`~networkx.Graph`)
        A Chimera or Pegasus graph. Either a nodes/edges pair
        or a NetworkX graph.

    Returns
    -------
    G : NetworkX Graph
        A Chimera or Pegasus working graph with the nodes and edges common to
        both input graphs.

    Examples
    ========

    This example creates a graph that represents a quarter (4 by 4 Chimera tiles)
    of a particular D-Wave system's working graph.

    >>> from dwave.system.samplers import DWaveSampler
    >>> sampler = DWaveSampler(solver={'qpu': True}) # doctest: +SKIP
    >>> C4 = dnx.chimera_graph(4)  # a 4x4 lattice of Chimera tiles
    >>> c4_working_graph = dnx.shared_working_graph(C4, [sampler.nodelist, sampler.edgelist])   # doctest: +SKIP
    """

    nodes1, edges1 = g1
    nodes2, edges2 = g2

    G1 = nx.Graph()
    G2 = nx.Graph()

    G1.add_nodes_from(nodes1)
    G1.add_edges_from(edges1)
    G2.add_nodes_from(nodes2)
    G2.add_edges_from(edges2)

    for node in nodes1:
        if node not in G2.nodes:
            G1.remove_node(node)

    for node in nodes2:
        if node not in G1.nodes:
            G2.remove_node(node)

    for node in G1.nodes:
        n_edges = list(G1.edges(node))
        for edge in n_edges:
            if edge not in G2.edges(node):
                G1.remove_edge(edge[0], edge[1])

    return(G1)
