# Copyright 2024 D-Wave
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


from enum import Enum as _Enum, auto as _auto

try:
    from enum import global_enum as _global_enum
except ImportError:
    # delete after Py3.10 EOL (2026/10)
    def _global_enum(cls):
        """This function is a specialized version of global_enum defined in py3.11

        We've omitted the functionality we aren't using, and kept the two pieces
        that are important.

        1. The members of an enum should be exported to the module containing
           them: `topology.ZEPHYR` is `topology.Topology.ZEPHYR`
        2. When `repr` is called on an enum member, it should look like
           'topology.ZEPHYR' rather than '<Topology.ZEPHYR: "zephyr">'
        """
        import sys

        toplevel = cls.__module__.split(".")[-1]
        cls.__repr__ = lambda self: f"{toplevel}.{self._name_}"
        # the following two lines are quoted directly from CPython.  It's hard
        # to imagine another way of writing this; hopefully this constitutes
        # fair use.
        sys.modules[cls.__module__].__dict__.update(cls.__members__)
        return cls


from dwave_networkx.utils.decorators import ImplementationHook as _ImplementationHook


class TopologyFamily:
    __slots__ = [
        "coordinates",
        "defect_free_graph",
        "draw",
        "draw_embedding",
        "draw_yield",
        "generator",
        "layout",
        "sublattice_mappings",
        "torus_generator",
    ]

    # Let's act like a string when we need to: str(), equality, and hash
    def __eq__(self, other):
        return other == self.value

    def __str__(self):
        return self.value.lower()

    def __hash__(self):
        return hash(self.value)

    def __init__(self, name):
        # DEV NOTE
        #
        # Throughout the codebase, we use the pattern
        #
        # @ZEPHYR.draw_embedding.implementation
        # def draw_zephyr_embedding(...):
        #     ...
        #
        # This is accomplished by using an ImplementationHook object.
        for slot in self.__slots__:
            setattr(self, slot, _ImplementationHook(self, slot))


class TopologyEnum(TopologyFamily, _Enum):
    @staticmethod
    def _generate_next_value_(name, *_):
        # make auto() work like in StrEnum
        return name.lower()


@_global_enum
class Topology(TopologyEnum):
    """An enumeration of qubit topologies supported by dwave_networkx.

    Each member of this enumeration functions as a string for the purpose of
    equality, hashing, and printing; and also contains a variety of utility
    functions specific to that topology.
    """

    CHIMERA = _auto()
    PEGASUS = _auto()
    ZEPHYR = _auto()


__all__ = ["Topology", *Topology.__members__]
