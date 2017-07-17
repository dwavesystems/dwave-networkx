"""
TODO
"""

from dwave_networkx.utils_qa.decorators import quantum_annealer_solver

__all__ = ["maximum_independent_set_qa"]


@quantum_annealer_solver(1)
def maximum_independent_set_qa(G, solver, **solver_args):
    """Tries to determine a maximum independent set of nodes using
    the provided quantum annealing (qa) solver.

    An independent set is a set of nodes such that the subgraph
    of G induced by these nodes contains no edges. A maximum
    independent set is an independent set of largest possible size.

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
    indep_nodes : list
       List of nodes that the form a maximum independent set, as
       determined by the given solver.

    Raises
    ------
    TypeError
        If the provided solver does not have a
        "solve_unstructured_qubo" method, an exception is raised.

    Examples
    --------
    >>> G = nx.path_graph(5)
    >>> dnx.maximum_independent_set_qa(G, solver)
    [0, 2, 4]

    Notes
    -----
    Quantum annealers by their nature do not necessarily return correct
    answers. This function makes no attempt to check the quality of the
    solution.

    https://en.wikipedia.org/wiki/Independent_set_(graph_theory)

    https://en.wikipedia.org/wiki/Quadratic_unconstrained_binary_optimization

    References
    ----------
    .. [1] Lucas, A. (2014). Ising formulations of many NP problems.
       Frontiers in Physics, Volume 2, Article 5.

    """

    # We assume that the solver can handle an unstructured QUBO problem, so let's set one up.
    # Let us define the largest independent set to be S.
    # For each node n in the graph, we assign a boolean variable v_n, where v_n = 1 when n
    # is in S and v_n = 0 otherwise.
    # We call the matrix defining our QUBO problem Q.
    # On the diagnonal, we assign the linear bias for each node to be -1. This means that each
    # node is biased towards being in S
    # On the off diagnonal, we assign the off-diagonal terms of Q to be 2. Thus, if both
    # nodes are in S, the overall energy is increased by 2.
    Q = {(node, node): -1 for node in G}
    Q.update({edge: 2 for edge in G.edges_iter()})

    # we expect that the solution will be a dict of the form {node: bool}
    response = solver.sample_qubo(Q, **solver_args)

    solution = next(response.samples())

    # nodes that are spin up or true are exactly the ones in S.
    return [node for node in solution if solution[node] > 0]
