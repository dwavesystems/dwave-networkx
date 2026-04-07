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

from enum import Enum
from functools import total_ordering

from dwave.graphs.generators.common.shape import TopologyShape
from dwave.graphs.tuplelike import TupleLike

__all__ = ["Coord", "CoordKind"]


class CoordKind(Enum):  # Kinds of coordinates of nodes in topologies
    CARTESIAN = 0
    TOPOLOGY = 1


@total_ordering
class Coord(TupleLike):
    """A class to represent the coordinate of a topology node."""

    __hash__ = TupleLike.__hash__

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

    @classmethod
    def kind(cls) -> CoordKind:
        """Returns the kind of the coordinate associated with the class.

        Raises:
            NotImplementedError: To be implemented in subclasses for
            specific coordinates.

        Returns:
            CoordKind: The kind of class's coordinate.
        """
        raise NotImplementedError

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def _args_valid_topology(self, *args, **kwargs) -> None:
        """Verifies the given coordinate is a valid coordinate."""
        raise NotImplementedError

    def is_shape_consistent(self, shape: TopologyShape) -> bool:
        """Tells whether the coordinate is consistent with a topology shape.

        Args:
            shape: The shape to check the consistency of the coordinate with.

        Returns:
            bool: Whether the coordinate is consistent with the shape.
        """
        raise NotImplementedError

    def is_quotient(self) -> bool:
        """Whether the given coordinate is a quotient coordinate."""
        raise NotImplementedError

    def to_quotient(
        self,
    ) -> Coord:
        """Converts the coordinate to its corresponding coordinate in a quotient graph."""
        raise NotImplementedError

    def to_non_quotient(
        self,
        shape: TopologyShape,
        **kwargs,
    ) -> list[Coord]:
        """Expands the coordinate to a non-quotient shape; i.e. it gives
        all coordinates in a non-quotient graph whose quotient is the coordinate.

        Args:
            shape: The non-quotient shape to expand the coordinate to.

        Returns:
            list[Coord]: The expansion of the coordinate into non-quotient.
        """
        raise NotImplementedError

    def convert(self, coord_kind: CoordKind) -> Coord:
        """Converts the coordinate to other kinds of coordinate in the same topology.

        Args:
            coord_kind (CoordKind): The coordinate kind to convert the coordinate to.

        Raises:
            NotImplementedError: To be implemented in subclasses for
            specific coordinates.

        Returns:
            Coord: The converted coordinate.
        """
        raise NotImplementedError

    def __eq__(self, other: object) -> bool:
        if (type(self) is not type(other)) or (self.is_quotient() != other.is_quotient()):
            return NotImplemented
        return self._tuple_format == other._tuple_format

    def __lt__(self, other: object) -> bool:
        if (type(self) is not type(other)) or (self.is_quotient() != other.is_quotient()):
            return NotImplemented
        return self._tuple_format < other._tuple_format
