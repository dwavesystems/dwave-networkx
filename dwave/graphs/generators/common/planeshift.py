# Copyright 2026 D-Wave
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
#
# ================================================================================================

from __future__ import annotations

from dwave.graphs.tuplelike import TupleLike

__all__ = ["TopologyPlaneShift"]


class TopologyPlaneShift(TupleLike):
    """Represents a displacement in a Cartesian plane associated with a topology.

    Args:
        x: The displacement in the x-direction of a Cartesian coordinate.
        y: The displacement in the y-direction of a Cartesian coordinate.
    """

    def __init__(self, x: int, y: int) -> None:
        self._check_args_valid(x, y)
        self._xy = (x, y)

    @classmethod
    def topology_name(cls) -> str:
        """Returns the name of the topology associated with the class.

        Raises:
            NotImplementedError: To be implemented in subclasses for
            specific topologies.

        Returns:
            str: The name of the topology the class is designed for.
        """
        raise NotImplementedError

    def _check_args_valid(self, x: int, y: int) -> None:
        """Verifies whether the given plane displacement is valid.

        Args:
            x: The displacement in the x-direction of a Cartesian coordinate.
            y: The displacement in the y-direction of a Cartesian coordinate.

        Raises:
            NotImplementedError: To be implemented in subclasses.
        """
        raise NotImplementedError

    @property
    def x(self) -> int:
        """The displacement in the x-direction"""
        return self._xy[0]

    @property
    def y(self) -> int:
        """The displacement in the y-direction"""
        return self._xy[1]

    def to_tuple(self) -> tuple[int, int]:
        """Returns the pair of values that uniquely identifies the displacement."""
        return self._xy

    @classmethod
    def _construct(cls: type[TopologyPlaneShift], *args) -> TopologyPlaneShift:
        """Constructs a class instance."""
        return cls(*args)

    def __mul__(self, scale: int) -> TopologyPlaneShift:
        """Multiplies the displacement from left by a scalar ``scale``.

        Args:
            scale: The scale for left-multiplying the displacement with.

        Returns:
            TopologyPlaneShift: The result of left-multiplying the displacement by ``scale``.
        """
        return self._construct(scale * self.x, scale * self.y)

    def __rmul__(self, scale: int) -> TopologyPlaneShift:
        """Multiplies the displacement from right by a scalar ``scale``.

        Args:
            scale: The scale for right-multiplying the displacement with.

        Returns:
            TopologyPlaneShift: The result of right-multiplying the displacement by ``scale``.
        """
        return self * scale

    def __add__(self, other: TopologyPlaneShift) -> TopologyPlaneShift:
        """
        Adds another displacement to the displacement.

        Args:
            other: The displacement to be added.

        Returns:
            TopologyPlaneShift: The displacement in Cartesian Coord by self followed by other.
        """
        return self._construct(self.x + other.x, self.y + other.y)
