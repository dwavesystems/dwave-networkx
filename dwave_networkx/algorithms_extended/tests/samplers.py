"""In order to test the discrete model functions, we need a discrete model
sampler. This module provides one.
"""
import itertools
import bisect

try:
    from dimod import SimulatedAnnealingSampler as FastSampler
except ImportError:
    FastSampler = None


class ExactSolver:
    """A very slow exact solver for QUBOs and Ising problems.

    Checks every possible combination, extremely slow in practice.
    """

    def sample_qubo(self, Q):
        """Brute force exact solver for QUBOs."""
        variables = set().union(*Q)

        response = []
        energies = []

        for ones in powerset(variables):
            sample = {v: v in ones and 1 or 0 for v in variables}
            energy = qubo_energy(Q, sample)

            idx = bisect.bisect(energies, energy)
            response.insert(idx, sample)
            energies.insert(idx, energy)

        return response

    def sample_ising(self, h, J):
        """Brute force exact solver for Ising problems."""

        response = []
        energies = []

        for ones in powerset(h):
            sample = {v: v in ones and 1 or -1 for v in h}
            energy = ising_energy(h, J, sample)

            idx = bisect.bisect(energies, energy)
            response.insert(idx, sample)
            energies.insert(idx, energy)

        return response


def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return itertools.chain.from_iterable(itertools.combinations(s, r) for r in range(len(s) + 1))


def qubo_energy(Q, sample):
    """Calculate the quadratic polynomial value of the given sample
    to a quadratic unconstrained binary optimization (QUBO) problem.
    """
    energy = 0

    for v0, v1 in Q:
        energy += sample[v0] * sample[v1] * Q[(v0, v1)]

    return energy


def ising_energy(h, J, sample):
    """Calculate the Ising energy of the given sample.
    """
    energy = 0

    # add the contribution from the linear biases
    for v in h:
        energy += h[v] * sample[v]

    # add the contribution from the quadratic biases
    for v0, v1 in J:
        energy += J[(v0, v1)] * sample[v0] * sample[v1]

    return energy
