"""
Makefile options for stanc and C++ compilers
"""

import os
from copy import copy
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from cmdstanpy.utils import get_logger

STANC_OPTS = [
    'O',
    'O0',
    'O1',
    'Oexperimental',
    'allow-undefined',
    'use-opencl',
    'warn-uninitialized',
    'include-paths',
    'name',
    'warn-pedantic',
]

STANC_DEPRECATED_OPTS = {
    'allow_undefined': 'allow-undefined',
    'include_paths': 'include-paths',
}

STANC_IGNORE_OPTS = [
    'debug-lex',
    'debug-parse',
    'debug-ast',
    'debug-decorated-ast',
    'debug-generate-data',
    'debug-mir',
    'debug-mir-pretty',
    'debug-optimized-mir',
    'debug-optimized-mir-pretty',
    'debug-transformed-mir',
    'debug-transformed-mir-pretty',
    'dump-stan-math-signatures',
    'auto-format',
    'print-canonical',
    'print-cpp',
    'o',
    'help',
    'version',
]

OptionalPath = Union[str, os.PathLike, None]


class CompilerOptions:
    """
    User-specified flags for stanc and C++ compiler.

    Attributes:
        stanc_options - stanc compiler flags, options
        cpp_options - makefile options (NAME=value)
        user_header - path to a user .hpp file to include during compilation
    """

    def __init__(
        self,
        *,
        stanc_options: Optional[Dict[str, Any]] = None,
        cpp_options: Optional[Dict[str, Any]] = None,
        user_header: OptionalPath = None,
    ) -> None:
        """Initialize object."""
        self._stanc_options = stanc_options if stanc_options is not None else {}
        self._cpp_options = cpp_options if cpp_options is not None else {}
        self._user_header = str(user_header) if user_header is not None else ''

    def __repr__(self) -> str:
        return 'stanc_options={}, cpp_options={}'.format(
            self._stanc_options, self._cpp_options
        )

    def __eq__(self, other: Any) -> bool:
        """Overrides the default implementation"""
        if self.is_empty() and other is None:  # equiv w/r/t compiler
            return True
        if not isinstance(other, CompilerOptions):
            return False
        return (
            self._stanc_options == other.stanc_options
            and self._cpp_options == other.cpp_options
            and self._user_header == other.user_header
        )

    def is_empty(self) -> bool:
        """True if no options specified."""
        return (
            self._stanc_options == {}
            and self._cpp_options == {}
            and self._user_header == ''
        )

    @property
    def stanc_options(self) -> Dict[str, Union[bool, int, str]]:
        """Stanc compiler options."""
        return self._stanc_options

    @property
    def cpp_options(self) -> Dict[str, Union[bool, int]]:
        """C++ compiler options."""
        return self._cpp_options

    @property
    def user_header(self) -> str:
        """user header."""
        return self._user_header

    def validate(self) -> None:
        """
        Check compiler args.
        Raise ValueError if invalid options are found.
        """
        self.validate_stanc_opts()
        self.validate_cpp_opts()
        self.validate_user_header()

    def validate_stanc_opts(self) -> None:
        """
        Check stanc compiler args and consistency between stanc and C++ options.
        Raise ValueError if bad config is found.
        """
        # pylint: disable=no-member
        if self._stanc_options is None:
            return
        ignore = []
        paths = None
        has_o_flag = False

        for deprecated, replacement in STANC_DEPRECATED_OPTS.items():
            if deprecated in self._stanc_options:
                if replacement:
                    get_logger().warning(
                        'compiler option "%s" is deprecated, use "%s" instead',
                        deprecated,
                        replacement,
                    )
                    self._stanc_options[replacement] = copy(
                        self._stanc_options[deprecated]
                    )
                    del self._stanc_options[deprecated]
                else:
                    get_logger().warning(
                        'compiler option "%s" is deprecated and '
                        'should not be used',
                        deprecated,
                    )
        for key, val in self._stanc_options.items():
            if key in STANC_IGNORE_OPTS:
                get_logger().info('ignoring compiler option: %s', key)
                ignore.append(key)
            elif key not in STANC_OPTS:
                raise ValueError(f'unknown stanc compiler option: {key}')
            elif key == 'include-paths':
                paths = val
                if isinstance(val, str):
                    paths = val.split(',')
                elif not isinstance(val, list):
                    raise ValueError(
                        'Invalid include-paths, expecting list or '
                        f'string, found type: {type(val)}.'
                    )
            elif key == 'use-opencl':
                if self._cpp_options is None:
                    self._cpp_options = {'STAN_OPENCL': 'TRUE'}
                else:
                    self._cpp_options['STAN_OPENCL'] = 'TRUE'
            elif key.startswith('O'):
                if has_o_flag:
                    get_logger().warning(
                        'More than one of (O, O1, O2, Oexperimental)'
                        'optimizations passed. Only the last one will'
                        'be used'
                    )
                else:
                    has_o_flag = True

        for opt in ignore:
            del self._stanc_options[opt]
        if paths is not None:
            bad_paths = [dir for dir in paths if not os.path.exists(dir)]
            if any(bad_paths):
                raise ValueError(
                    'invalid include paths: {}'.format(', '.join(bad_paths))
                )

            self._stanc_options['include-paths'] = [
                os.path.abspath(os.path.expanduser(path)) for path in paths
            ]

    def validate_cpp_opts(self) -> None:
        """
        Check cpp compiler args.
        Raise ValueError if bad config is found.
        """
        if self._cpp_options is None:
            return
        for key in ['OPENCL_DEVICE_ID', 'OPENCL_PLATFORM_ID']:
            if key in self._cpp_options:
                self._cpp_options['STAN_OPENCL'] = 'TRUE'
                val = self._cpp_options[key]
                if not isinstance(val, int) or val < 0:
                    raise ValueError(
                        f'{key} must be a non-negative integer value,'
                        f' found {val}.'
                    )

    def validate_user_header(self) -> None:
        """
        User header exists.
        Raise ValueError if bad config is found.
        """
        if self._user_header != "":
            if not (
                os.path.exists(self._user_header)
                and os.path.isfile(self._user_header)
            ):
                raise ValueError(
                    f"User header file {self._user_header} cannot be found"
                )
            if self._user_header[-4:] != '.hpp':
                raise ValueError(
                    f"Header file must end in .hpp, got {self._user_header}"
                )
            if "allow-undefined" not in self._stanc_options:
                self._stanc_options["allow-undefined"] = True
            # set full path
            self._user_header = os.path.abspath(self._user_header)

            if ' ' in self._user_header:
                raise ValueError(
                    "User header must be in a location with no spaces in path!"
                )

            if (
                'USER_HEADER' in self._cpp_options
                and self._user_header != self._cpp_options['USER_HEADER']
            ):
                raise ValueError(
                    "Disagreement in user_header C++ options found!\n"
                    f"{self._user_header}, {self._cpp_options['USER_HEADER']}"
                )

            self._cpp_options['USER_HEADER'] = self._user_header

    def add(self, new_opts: "CompilerOptions") -> None:  # noqa: disable=Q000
        """Adds options to existing set of compiler options."""
        if new_opts.stanc_options is not None:
            if self._stanc_options is None:
                self._stanc_options = new_opts.stanc_options
            else:
                for key, val in new_opts.stanc_options.items():
                    if key == 'include-paths':
                        self.add_include_path(str(val))
                    else:
                        self._stanc_options[key] = val
        if new_opts.cpp_options is not None:
            for key, val in new_opts.cpp_options.items():
                self._cpp_options[key] = val
        if new_opts._user_header != '' and self._user_header == '':
            self._user_header = new_opts._user_header

    def add_include_path(self, path: str) -> None:
        """Adds include path to existing set of compiler options."""
        path = os.path.abspath(os.path.expanduser(path))
        if 'include-paths' not in self._stanc_options:
            self._stanc_options['include-paths'] = [path]
        elif path not in self._stanc_options['include-paths']:
            self._stanc_options['include-paths'].append(path)

    def compose_stanc(self) -> List[str]:
        opts = []
        if self._stanc_options is not None and len(self._stanc_options) > 0:
            for key, val in self._stanc_options.items():
                if key == 'include-paths':
                    opts.append(
                        '--include-paths='
                        + ','.join(
                            (
                                Path(p).as_posix()
                                for p in self._stanc_options['include-paths']
                            )
                        )
                    )
                elif key == 'name':
                    opts.append(f'--name={val}')
                else:
                    opts.append(f'--{key}')
        return opts

    def compose(self) -> List[str]:
        """Format makefile options as list of strings."""
        opts = ['STANCFLAGS+=' + flag for flag in self.compose_stanc()]
        if self._cpp_options is not None and len(self._cpp_options) > 0:
            for key, val in self._cpp_options.items():
                opts.append(f'{key}={val}')
        return opts
