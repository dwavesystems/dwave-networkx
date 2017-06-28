"""
TODO
"""

import sys
import math
import itertools

import dwave_networkx as dnx
from dwave_networkx.utils_qa.decorators import quantum_annealer_solver
from dwave_networkx.exception import DWaveNetworkXQAException

__all__ = ["min_vertex_coloring_qa"]

# compatibility for python 2/3
if sys.version_info[0] == 2:
    range = xrange


@quantum_annealer_solver(1)
def min_vertex_coloring_qa(G, solver, **solver_args):
    """TODO

    https://en.wikipedia.org/wiki/Brooks%27_theorem
    """

    N = len(G)  # number of nodes
    M = len(G.edges())  # number of edges

    # ok, first up, we can eliminate a few graph types trivially

    # Graphs with no edges, have chromatic number 1
    if not M:
        return {node: 0 for node in G}

    # Complete graphs have chromatic number N
    if M == N * (N - 1) / 2:
        return {node: color for color, node in enumerate(G)}

    # The number of variables in the QUBO is approximately the number of nodes in the graph
    # times the number of potential colors, so we want as tight an upper bound on the
    # chromatic number (chi) as possible

    # we know that chi <= max degree, unless it is complete or a cycle graph, in which
    # case chi <= max degree + 1 (Brook's Theorem)
    if N % 2 == 1 and is_cycle(G):
        # odd cycle graphs need three colors
        ub = 3
    else:
        ub = max(G.degree(node) for node in G)

    # we also know that chi*(chi-1) <= 2*|E|, so using the quadratic formula
    ub = min(ub, int(math.ceil((1 + math.sqrt(1 + 8 * M)) / 2)))

    # now we can start coloring. We start by finding a clique. Without loss of
    # generality, we can color all of the nodes in the clique with unique colors.
    # This also gives us a lower bound on the number of colors we need
    clique = _greedy_clique_about_node(G, next(G.nodes_iter()))
    coloring = {node: idx for idx, node in enumerate(clique)}
    lb = len(clique) - 1

    # ok, now that we have an upper bound, it is time to start building the QUBO. For each
    # node n, and for each color c, we define a boolean variable v_n_c that is 1 when node
    # n is color c, and 0 otherwise.

    # We assign each variable an integer label
    counter = itertools.count()
    qubo_variables = {}
    for node in G:
        if node not in coloring:
            impossible_colors = {coloring[n] for n in G[node] if n in coloring}

            qubo_variables[node] = {color: next(counter) for color in range(ub)
                                    if color not in impossible_colors}

    # now we have three different constraints we wish to add.
    Q = {}

    # first, we want neighboring nodes to have different colors. To this we add a penalty
    # in the QUBO when v_n0_c and v_n1_c
    for n0, n1 in itertools.combinations(qubo_variables, 2):
        if n0 not in G[n1]:
            # we can ignore these if they are not neighbors
            continue
        for color in qubo_variables[n0]:
            if color in qubo_variables[n1]:
                idx0 = qubo_variables[n0][color]
                idx1 = qubo_variables[n1][color]

                Q[(idx0, idx1)] = 1

    # second, we want each node to have exactly one color
    for node in qubo_variables:
        for color in qubo_variables[node]:
            idx = qubo_variables[node][color]
            Q[(idx, idx)] = -1

        for c0, c1 in itertools.combinations(qubo_variables[node], 2):
            idx0 = qubo_variables[node][c0]
            idx1 = qubo_variables[node][c1]

            Q[(idx0, idx1)] = 2

    # third, since we want the minimum vertex color, we want to disincentivize
    # the colors we have not already used. In increasing amounts
    for c in range(lb + 1, ub + 1):
        penalty = .5 * (c - lb) / (ub - lb)

        print c, penalty

        for node in qubo_variables:
            if c in qubo_variables[node]:
                idx = qubo_variables[node][c]
                Q[(idx, idx)] += penalty

    # ok, let's solve
    print qubo_variables
    print Q
    solution = solver.solve_qubo(Q, **solver_args)

    print solution


    for node in qubo_variables:
        for color in qubo_variables[node]:
            idx = qubo_variables[node][color]

            if solution[idx]:
                coloring[node] = color
                break

        if node not in coloring:
            raise DWaveNetworkXQAException("node {} did not recieve a color".format(node))

    # finally return the coloring
    return coloring


def _greedy_clique_about_node(G, n):
    """Really simple attempt to find the largest clique containing node n"""
    H = G.subgraph(G[n])
    if not H:
        return [n]
    return [n] + _greedy_clique_about_node(H, next(H.nodes_iter()))


def is_cycle(G):
    """Determines whether the given graph is a connected cycle graph.

    https://en.wikipedia.org/wiki/Cycle_graph
    """
    trailing, leading = next(iter(G.edges()))
    start_node = trailing

    # travel around the graph, checking that each node has degree exactly two
    # also track how many nodes were visted
    n_visited = 1
    while leading != start_node:
        neighbors = G[leading]

        if len(neighbors) != 2:
            return False

        node1, node2 = neighbors

        if node1 == trailing:
            trailing, leading = leading, node2
        else:
            trailing, leading = leading, node1

        n_visited += 1

    # if we havent visited all of the nodes, then it is not a connected cycle
    return n_visited == len(G)
