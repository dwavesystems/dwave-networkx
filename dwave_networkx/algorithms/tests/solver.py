# we need a test solver
_solver_found = True
try:
    from dwave_sapi2.local import local_connection
    from dwave_sapi2.core import solve_ising, solve_qubo
    from dwave_sapi2.util import get_hardware_adjacency, qubo_to_ising
    from dwave_sapi2.embedding import find_embedding, embed_problem, unembed_answer
except ImportError as e:
    _solver_found = False


class Solver(object):
    """These qa functions all assume that there is a solver that can handle them
    This is quick-and-dirty solver that wraps the sapi software solver.
    """
    def solve_qubo(self, Q, **args):
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

        # next let's make Q upper triangular
        Qut = {}
        for (n0, n1), val in Q.items():
            if n1 < n0:
                n0, n1 = n1, n0

            if (n0, n1) not in Qut:
                Qut[(n0, n1)] = val
            else:
                Qut[(n0, n1)] += val

        # get the solfware solver from sapi
        solver = local_connection.get_solver("c4-sw_sample")
        A = get_hardware_adjacency(solver)

        # convert the problem to Ising
        (h, J, ising_offset) = qubo_to_ising(Qut)

        if not J:
            ans = [[bias > 0 and -1 or 1 for bias in h]]
        else:
            # get the embedding, this function assumes that the given problem is
            # unstructured
            embeddings = find_embedding(J, A, tries=50)
            if not embeddings:
                raise Exception('problem is too large to be embedded')
            [h0, j0, jc, embeddings] = embed_problem(h, J, embeddings, A)

            # actually solve the thing
            j = j0
            j.update(jc)
            result = solve_ising(solver, h0, j, num_reads=10)
            ans = unembed_answer(result['solutions'], embeddings, 'minimize_energy', h, J)

        ground = min(result['energies'])

        # unapply the relabelling and convert back from spin
        inv_label = {label[n]: n for n in label}
        return [{inv_label[i]: (spin + 1) / 2 for i, spin in enumerate(a)}
                for idx, a in enumerate(ans) if result['energies'][idx] <= ground + .0000000001]
