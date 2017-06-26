"""
TODO
"""

__all__ = ['min_vertex_cover_qa']


def min_vertex_cover_qa(G, solver, **solver_args):
    """TODO
    """

    if not hasattr(solver, "solve_unstructured_qubo") \
            or not callable(solver.solve_unstructured_qubo):
        raise TypeError("expected solver to have a 'solve_unstructured_qubo' method")

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
    Q = {(node, node): B - A*G.degree(node) for node in G}
    Q.update({edge: A for edge in G.edges_iter()})

    # we expect that the solution will be a dict of the form {node: bool}
    solution = solver.solve_unstructured_qubo(Q, **solver_args)

    # nodes that are true are in the cover
    return [node for node in G if solution[node] > 0]
