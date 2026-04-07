from functools import cached_property
from typing import Any, Iterator


__all__ = ["TupleLike"]


class TupleLike:
    """A blueprint class to represent objects which can be identified
        with a fixed number of ordered values."""
    def __init__(self, *args, **kwargs) -> None:
        pass
    
    def to_tuple(self) -> tuple[Any, ...]:
        """Returns the tuple of values that uniquely identifies the object.

        Raises:
            NotImplementedError: To be implemented in subclasses.

        Returns:
            tuple[Any, ...]: The tuple of values that uniquely identifies the object.
        """
        raise NotImplementedError

    @cached_property
    def _tuple_format(self) -> tuple[Any, ...]:
        """The tuple associated with the object."""
        return self.to_tuple()

    def __eq__(self, value: object) -> bool:
        if type(self) is not type(value):
            return NotImplemented
        return self._tuple_format == value._tuple_format

    def __hash__(self) -> int:
        return hash((type(self), self._tuple_format))

    def __iter__(self) -> Iterator[Any]:
        return iter(self._tuple_format)

    def __len__(self) -> int:
        return len(self._tuple_format)

    def __getitem__(self, i: int) -> Any:
        return self._tuple_format[i]

    def __repr__(self) -> str:
        return f"{type(self).__name__}{self._tuple_format}"

    def __str__(self) -> str:
        return f"{self._tuple_format}"
