"""
Temporary helpers for deprecations in the 1.x series.
"""

import functools
import inspect
from typing import Any, Callable, TypeVar

from .logging import get_logger

F = TypeVar('F', bound=Callable[..., Any])


def deprecate_positional_args(last_deprecated: str) -> Callable[[F], F]:
    """
    Allow keyword-only arguments up to and including ``last_deprecated``
    to still be passed by position, logging a deprecation warning.

    Positional use of these arguments becomes an error in CmdStanPy 2.0.
    """

    def decorator(func: F) -> F:
        params = inspect.signature(func).parameters.values()
        n_positional = sum(
            p.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD for p in params
        )
        kw_only = [
            p.name for p in params if p.kind == inspect.Parameter.KEYWORD_ONLY
        ]
        deprecated = kw_only[: kw_only.index(last_deprecated) + 1]

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            extra = args[n_positional:]
            if not extra or len(extra) > len(deprecated):
                # either nothing to do, or let Python raise its usual error
                return func(*args, **kwargs)
            names = deprecated[: len(extra)]
            for name in names:
                if name in kwargs:
                    raise TypeError(
                        f"{func.__qualname__}() got multiple values for "
                        f"argument '{name}'"
                    )
            quoted = ", ".join("`" + name + "`" for name in names)
            get_logger().warning(
                "Passing %s to %s() by position is deprecated and will raise "
                "an error in CmdStanPy 2.0. Please pass them as keyword "
                "arguments instead.",
                quoted,
                func.__qualname__,
            )
            kwargs.update(zip(names, extra))
            return func(*args[:n_positional], **kwargs)

        return wrapper  # type: ignore[return-value]

    return decorator
