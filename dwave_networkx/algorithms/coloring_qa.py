"""
TODO
"""

import math
import itertools

import dwave_networkx as dnx

__all__ = ["min_vertex_coloring_qa"]


def min_vertex_coloring_qa(G, solver, **solver_args):
    """TODO

    https://en.wikipedia.org/wiki/Brooks%27_theorem
    """

    if not hasattr(solver, "solve_unstructured_qubo") \
            or not callable(solver.solve_unstructured_qubo):
        raise TypeError("expected solver to have a 'solve_unstructured_qubo' method")

    N = len(G)  # number of nodes
    M = len(G.edges())  # number of edges

    # ok, first up, we can eliminate a few graph types trivially

    # Graphs with no edges, have chromatic number 1
    if not M:
        return {node: 0 for node in G}

    # Complete graphs have chromatic number N
    if M == N*(N-1)/2:
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
    ub = min(ub, int((1 + math.sqrt(1 + 8*M))/2))

    # before we get started building the QUBO, we can reduce the problem size by assigning
    # some colors directly. Specifically, for a random edge, we can assign one node color 0
    # and its neighbor color 1
    n0, n1 = next(G.edges_iter())
    coloring = {n0: 0, n1: 1}
    lb = 2  # lower bound on the number of colors needed

    # now we might as well see if we're already in a clique. Each node in the clique gets its
    # own color
    shared = reduce(set.intersection, [set(G[node]) for node in coloring])
    while shared:
        node = shared.pop()
        lb += 1
        coloring[node] = lb
        shared = reduce(set.intersection, [set(G[node]) for node in coloring])

    # Ok, we have everything we need to start building the QUBO! For each node n in the graph
    # and for each color c, we need a boolean variable v_n_c which is 1 when node n is colored
    # c and 0 otherwise.
    qubo_variables = {node: {color: 'v_{}_{}'.format(node, color) for color in range(ub)}
                      for node in G}

    # ok, for the nodes that are already colored, we can remove all of their associated
    # variables, and we can remove the variables associated with their color for all
    # of their neighbors
    for node in coloring:
        qubo_variables[node] = {}

        color = coloring[node]
        for neighbor in G[node]:
            if color in qubo_variables[neighbor]:
                del qubo_variables[neighbor][color]

    # # finally it is possible that some have only one possible color, so let's go ahead and
    # # remove those
    # while any(len(colors) == 1 for colors in qubo_variables.values()):
    #     for node in G:
    #         if len(qubo_variables[node]) == 1:
    #             color = next(iter(qubo_variables[node]))
    #             coloring[node] = color

    #             qubo_variables[node] = {}
    #             for neighbor in G[node]:
    #                 if color in qubo_variables[neighbor]:
    #                     del qubo_variables[neighbor][color]

    # it is possible that we are already done
    if all(node in coloring for node in G):
        return coloring

    # Now with all of the variables set, we can actually start building the qubo
    Q = {}

    # First, for each two nodes that are neighbors, we need that their color is not equal
    for n0, n1 in G.edges_iter():
        for color in qubo_variables[n0]:
            if color in qubo_variables[n1]:
                Q[(qubo_variables[n0][color], qubo_variables[n1][color])] = 2

    # now for each node, we require that they get assigned exactly one color
    for node in G:
        # disincentize having more than one color
        for color1, color2 in itertools.combinations(qubo_variables[node], 2):
            Q[(qubo_variables[node][color1], qubo_variables[node][color2])] = 2

        # incentivize colors being present
        for color in qubo_variables[node]:
            Q[(qubo_variables[node][color], qubo_variables[node][color])] = -1

    # # finally, we want to disincentivize colors beyond the lower bound, by adding a small h bias to
    # # them
    # for color in range(lb, ub):
    #     for node in G:
    #         if color in indices[node]:
    #             Q[(indices[node][color], indices[node][color])] += 1.*(color-lb+1)/(ub - lb + 1)

    if not Q:
        print coloring
        print G.nodes()
        print G.edges()
        print qubo_variables
        raise Exception

    solution = solver.solve_unstructured_qubo(Q, **solver_args)

    # decode the solution by reading off the color variables
    for node in G:
        if node in coloring:
            continue

        for color in qubo_variables[node]:
            var = qubo_variables[node][color]
            if solution[var] > 0:
                coloring[node] = color
                continue

    if not all(node in coloring for node in G):
        raise dnx.DWaveNetworkXQAException('Not every node was assigned a color.')

    # finally return the coloring
    return coloring


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
