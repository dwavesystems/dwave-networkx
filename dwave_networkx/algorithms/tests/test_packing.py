import unittest

import dwave_networkx as dnx

# we need a test solver
from dwave_sapi2.local import local_connection
from dwave_sapi2.core import solve_ising, solve_qubo
from dwave_sapi2.util import get_chimera_adjacency, qubo_to_ising
from dwave_sapi2.embedding import find_embedding, embed_problem, unembed_answer


class Solver(object):
    """These qa functions all assume that there is a solver that can handle them
    This is quick-and-dirty solver that wraps the sapi software solver.
    """
    def solve_unstructured_qubo(self, Q, **args):
        # relabel Q with indices
        label = {}
        idx = 0
        for n1, n2 in Q:
            if n1 not in label:
                label[n1] = idx
                idx += 1
            if n2 not in label:
                label[n2] = idx
                idx += 1
        Qrl = {(label[n1], label[n2]): Q[(n1, n2)] for (n1, n2) in Q}

        # get the solfware solver from sapi
        solver = local_connection.get_solver("c4-sw_optimize")
        A = get_chimera_adjacency(4, 4, 4)

        # convert the problem to Ising
        (h, J, ising_offset) = qubo_to_ising(Qrl)

        # get the embedding, this function assumes that the given problem is
        # unstructured
        embeddings = find_embedding(Qrl, A)
        [h0, j0, jc, embeddings] = embed_problem(h, J, embeddings, A)

        # actually solve the thing
        j = j0
        j.update(jc)
        result = solve_ising(solver, h0, j)
        ans = unembed_answer(result['solutions'], embeddings, 'minimize_energy', h, J)

        # unapply the relabelling and return
        inv_label = {label[n]: n for n in label}
        return {inv_label[i]: spin for i, spin in enumerate(ans[0])}


#######################################################################################
# Unit Tests
#######################################################################################

class TestPacking(unittest.TestCase):
    def test_maximum_independent_set_basic(self):
        """Runs the function on some small and simple graphs, just to make
        sure it works in basic functionality.
        """

        G = dnx.chimera_graph(2, 2, 4)

        indep_set = dnx.maximum_independent_set_qa(G, Solver())
        self.set_independence_check(G, indep_set)

        G = dnx.path_graph(5)
        indep_set = dnx.maximum_independent_set_qa(G, Solver())
        self.set_independence_check(G, indep_set)

#######################################################################################
# Helper functions
#######################################################################################

    def set_independence_check(self, G, indep_set):
        """Check that the given set of nodes are in fact nodes in the graph and
        independent of eachother (that is there are no edges between them"""
        subG = G.subgraph(indep_set)
        self.assertTrue(len(subG.edges()) == 0)
