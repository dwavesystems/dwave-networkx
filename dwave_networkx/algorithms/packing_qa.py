"""
TODO
"""

__all__ = ["maximum_independent_set_qa"]


def maximum_independent_set_qa(G, solver, **solver_args):
    """Tries to determine a maximum independent set of nodes using
    the provided quantum annealing (qa) solver.

    An independent set is a set of nodes such that the subgraph
    of G induced by these nodes contains no edges. A maximum
    independent set and independent set of largest possible size.

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
    Quantum annealers by their nature to not necessarily return correct
    answers. This function makes no attempt to check the quality of the
    solution.

    References
    ----------
    .. [1] Lucas, A. (2014). Ising formulations of many NP problems.
       Frontiers in Physics, Volume 2, Article 5.

    """

    if not hasattr(solver, "solve_unstructured_qubo") \
            or not callable(solver.solve_unstructured_qubo):
        raise TypeError("expected solver to have a 'solve_unstructured_qubo' method")

    # We assume that the solver can handle an unstructured QUBO problem, so let's set one up.
    Q = {(node, node): -1 for node in G}
    Q.update({edge: 2 for edge in G.edges_iter()})

    # we expect that the solution will be a dict of the form {node: spin} where each spin = +/-1
    solution = solver.solve_unstructured_qubo(Q, **solver_args)

    return [node for node in solution if solution[node] > 0]
