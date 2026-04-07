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


from typing import Literal

from dwave.graphs.generators.common import TopologyPlaneShift

__all__ = ["ZephyrPlaneShift"]


class ZephyrPlaneShift(TopologyPlaneShift):
    """Represents a displacement in the Zephyr quotient plane (expressed in Cartesian coordinates).

    Args:
        x: The displacement in the x-direction of a Zephyr Cartesian coordinate.
        y: The displacement in the y-direction of a Zephyr Cartesian coordinate.

    Raises:
        ValueError: If ``x`` and ``y`` have different parity.
    """

    @classmethod
    def topology_name(cls) -> Literal["zephyr"]:
        """Returns the name of the topology associated with the class."""
        return "zephyr"

    def _check_args_valid(self, x: int, y: int) -> None:
        """Verifies whether the given plane displacement is valid.

        Args:
            x: The displacement in the x-direction of a Cartesian coordinate.
            y: The displacement in the y-direction of a Cartesian coordinate.

        Raises:
            ValueError: If ``x`` and ``y`` have different parity.
        """
        if x % 2 != y % 2:
            raise ValueError(f"Expected x, y to have the same parity, got {x, y}")
