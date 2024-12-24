# delete after Py3.10 is dropped
# The implementation of StrEnum and global_enum are strongly informed by
# those found in Py3.11 but are significantly less general.
    
from enum import Enum as _Enum
import sys as _sys

class StrEnum(str, _Enum):
    def __new__(cls, value):
        member = str.__new__(cls, value)
        member._value_ = value
        return member

    @staticmethod
    def _generate_next_value_(name, *_):
        #make auto() work
        return name.lower()

    def __str__(self):
        return self._value_

def global_enum(cls):
    toplevel = cls.__module__.split('.')[-1]
    cls.__repr__ = lambda self: f'{toplevel}.{self._name_}'
    _sys.modules[cls.__module__].__dict__.update(cls.__members__)
    return cls

