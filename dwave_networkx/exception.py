from dwave_networkx import NetworkXException


class DWaveNetworkXException(NetworkXException):
    """Base class for exceptions in DWaveNetworkX"""


class DWaveNetworkXQAException(DWaveNetworkXException):
    """Exception that is raised when a quantum annealer has returned an invalid solution.

    Because quantum annealers have a non-zero probability of returning an excited state,
    it is important to check the returned solution for correctness. In the case that the
    returned solution is not correct, this exception is raised.
    """
