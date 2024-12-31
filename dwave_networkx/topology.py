from enum import Enum, auto

try:
    from enum import global_enum
except ImportError:
    # delete after Py3.10 EOL (2026/10)
    import sys

    def global_enum(cls):
        toplevel = cls.__module__.split(".")[-1]
        cls.__repr__ = lambda self: f"{toplevel}.{self._name_}"
        sys.modules[cls.__module__].__dict__.update(cls.__members__)
        return cls


from .utils.decorators import ImplementationHook


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

    def _install_dispatch(self, name):
        return partial(setattr, self, name)

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
            setattr(self, slot, ImplementationHook(self, slot))


class TopologyEnum(TopologyFamily, Enum):
    @staticmethod
    def _generate_next_value_(name, *_):
        # make auto() work like in StrEnum
        return name.lower()


@global_enum
class Topology(TopologyEnum):
    """An enumeration of qubit topologies supported by dwave_networkx.

    Each member of this enumeration functions as a string for the purpose of
    equality, hashing, and printing; and also contains a variety of utility
    functions specific to that topology.
    """

    CHIMERA = auto()
    PEGASUS = auto()
    ZEPHYR = auto()


__all__ = ["Topology", *Topology.__members__]
