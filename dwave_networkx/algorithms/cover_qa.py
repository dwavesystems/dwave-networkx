"""
TODO
"""

from dwave_networkx.utils_qa.decorators import quantum_annealer_solver

__all__ = ['min_vertex_cover_qa']


@quantum_annealer_solver(1)
def min_vertex_cover_qa(G, solver, **solver_args):
    """Tries to determine a minimum vertex cover using the provided
    quantum annealing (qa) solver.

    A vertex cover is a set of verticies such that each edge of the graph
    is incident with at least one vertex in the set. A minimum vertex cover
    is the vertex cover of smallest size.

    Parameters
    ----------
    G : NetworkX graph

    solver
        A quantum annealing solver. Expectes the solver to have a
        "solve_unstructured_qubo" method that returns a single
        solution of the form {node: spin, ...}.

    Additional keyword parameters are passed to the given solver.

    Returns
    -------
    vertex_cover : list
       List of nodes that the form a the minimum vertex cover, as
       determined by the given solver.

    Raises
    ------
    TypeError
        If the provided solver does not have a
        "solve_unstructured_qubo" method, an exception is raised.

    Examples
    --------
    >>> G = dnx.chimera_graph(2, 2, 4)
    >>> dnx.min_vertex_cover_qa(G, Solver())
    [0, 1, 2, 3, 12, 13, 14, 15, 20, 21, 22, 23, 24, 25, 26, 27]

    Notes
    -----
    Quantum annealers by their nature do not necessarily return correct
    answers. This function makes no attempt to check the quality of the
    solution.

    https://en.wikipedia.org/wiki/Vertex_cover

    https://en.wikipedia.org/wiki/Quadratic_unconstrained_binary_optimization

    References
    ----------
    .. [1] Lucas, A. (2014). Ising formulations of many NP problems.
       Frontiers in Physics, Volume 2, Article 5.

    """

    # our weights for the two components. We need B < A
    A = 1  # term for ensuring that each edge has at least one node
    B = .5  # term for minimizing the number of nodes colored

    # ok, let's build the qubo. For each node n in the graph we assign a boolean variable
    # v_n where v_n = 1 when n is part of the vertex cover and v_n = 0 when n is not.

    # For each edge, we want at least one node to be colored. That is, for each edge
    # (n1, n2) we want a term in the hamiltonian A*(1 - v_n1)*(1 - v_n2). This can
    # be rewritten A*(1 - v_n1 - v_n2 + v_n1*v_n2). Since each edge contributes one
    # of these, our final hamiltonian is
    # H_a = A*(|E| + sum_(n) -(deg(n)*v_n) + sum_(n1,n2) v_n1*v_n2)
    # additionally, we want to have the minimum cover, so we add H_b = B*sum_(n) v_n
    Q = {(node, node): B - A * G.degree(node) for node in G}
    Q.update({edge: A for edge in G.edges_iter()})

    # we expect that the solution will be a dict of the form {node: int(bool)}
    solution = solver.solve_qubo(Q, **solver_args)

    # nodes that are true are in the cover
    return [node for node in G if solution[node] > 0]
