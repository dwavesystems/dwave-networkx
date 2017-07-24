from dwave_networkx import NetworkXException


class DWaveNetworkXException(NetworkXException):
    """Base class for exceptions in DWaveNetworkX."""


class DWaveNetworkXMissingSampler(DWaveNetworkXException):
    """No sampler provided."""
