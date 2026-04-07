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

from typing import Callable, Iterable

from dwave.graphs.generators.common.coord import Coord, CoordKind
from dwave.graphs.generators.common.node_edge import EdgeKind, TopologyEdge, TopologyNode
from dwave.graphs.generators.common.planeshift import TopologyPlaneShift
from dwave.graphs.generators.common.shape import TopologyShape

__all__ = ["Topology"]


class Topology:
    def __init__(
        self,
        planeshift_class: TopologyPlaneShift,
        shape_class: TopologyShape,
        node_class: TopologyNode,
        edge_class: TopologyEdge,
        coord_class: Coord | Iterable[Coord],
    ) -> None:
        """A class to access various classes associated with a topology.

        Args:
            planeshift_class: The displacement class in a Cartesian plane associated with the topology.
            shape_class: The shape class associated with the topology.
            node_class: The node class associated with the topology.
            edge_class: The edge class associated with the topology.
            coord_class: The coordinate class(es) associated with the topology.

        Raises:
            ValueError: If the topology names of the arguments are not consistent.
        """
        _coord_classes = [coord_class] if isinstance(coord_class, Coord) else coord_class
        all_topology_names = {
            _associated_class_.topology_name()
            for _associated_class_ in (planeshift_class, shape_class, node_class, edge_class)
        } | {_associated_class_.topology_name() for _associated_class_ in _coord_classes}
        if len(all_topology_names) != 1:
            raise ValueError(
                f"All arguments must have the same topology_name, got {all_topology_names}"
            )
        self._check_node_edge_class(
            node_class=node_class,
            edge_class=edge_class,
        )
        self.planeshift_class = planeshift_class
        self.shape_class = shape_class
        self.coord_class = _coord_classes
        self.node_class = node_class
        self.edge_class = edge_class

    @classmethod
    def _check_node_edge_class(cls, node_class: TopologyNode, edge_class: TopologyEdge):
        """Checks whether the node and edge classes are consistent.

        Args:
            node_class: The node class associated with the topology.
            edge_class: The edge class associated with the topology.

        Raises:
            ValueError: If the edge class's associated node class is not the given node class.
            ValueError: If the node class's associated edge class is not the given edge class.
        """
        if edge_class.associated_topology_node is not node_class:
            raise ValueError(
                f"The node_class doen't match with the associated node class for {edge_class}"
            )
        if node_class.associated_topology_edge is not edge_class:
            raise ValueError(
                f"The edge_class doen't match with the associated edge class for {node_class}"
            )

    def nodes(
        self,
        shape: TopologyShape,
        coord_kind: CoordKind | None = None,
    ) -> set[TopologyNode]:
        """Returns the nodes of the topology graph with a shape.

        Args:
            shape: The shape of topology graph.
            coord_kind: The kind of coordinate the nodes are represented with.
                Defaults to ``None``.
        Raises:
            NotImplementedError: To be implemented in subclasses.

        Returns:
            set[TopologyNode]: The nodes of the topology graph with the given
                shape and coordinate kind.
        """
        raise NotImplementedError

    def edges(
        self,
        shape: TopologyShape,
        edge_kind: EdgeKind | Iterable[EdgeKind] | None = None,
        where: Callable[[Coord], bool] = lambda coord: True,
        coord_kind: CoordKind | None = None,
    ) -> set[TopologyEdge]:
        """Returns the edges of the topology graph with a shape and optional
            coordinate and edge kind.

        Args:
            shape: The shape of topology graph.
            edge_kind:
                Edge kind filter. Restricts edges to the given edge kind(s).
                If ``None``, no filtering is applied. Defaults to ``None``.
            where: A coordinate filter. Defaults to always.
            coord_kind:
                The kind of coordinate the edges endpoints are represented with.
                Defaults to ``None``.

        Raises:
            NotImplementedError: To be implemented in subclasses.

        Returns:
            set[TopologyEdge]: The edges of the topology graph with the given
                shape, coordinate and edge kind.
        """
        raise NotImplementedError
