"""
Base exceptions and errors for D-Wave NetworkX.

All are derived from NetworkXException.

"""

from networkx import NetworkXException


class DWaveNetworkXException(NetworkXException):
    """Base class for exceptions in DWaveNetworkX."""


class DWaveNetworkXMissingSampler(DWaveNetworkXException):
    """Exception raised by an algorithm requiring a discrete model
    sampler when none is provided."""
