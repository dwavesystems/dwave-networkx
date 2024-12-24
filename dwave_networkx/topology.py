try:
    from enum import StrEnum as _StrEnum, global_enum as _global_enum, auto as _auto
except ImportError:
    from enum import auto as _auto
    from .__future_enum import StrEnum as _StrEnum, global_enum as _global_enum

@_global_enum
class Topology(_StrEnum):
    CHIMERA = _auto()
    PEGASUS = _auto()
    ZEPHYR = _auto()
    
__all__ = ['Topology', *Topology.__members__]
