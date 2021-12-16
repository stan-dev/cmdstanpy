"""Utilities for the argument classes."""

from enum import Enum, auto


class Method(Enum):
    """Supported CmdStan method names."""

    SAMPLE = auto()
    OPTIMIZE = auto()
    GENERATE_QUANTITIES = auto()
    VARIATIONAL = auto()

    def __repr__(self) -> str:
        return '<%s.%s>' % (self.__class__.__name__, self.name)
