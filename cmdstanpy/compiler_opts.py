"""
Makefile options for stanc and C++ compilers
"""

import os
import logging

from pathlib import Path
from typing import Dict, List

from cmdstanpy.utils import get_logger

STANC_OPTS = [
    'O',
    'allow_undefined',
    'use-opencl',
    'warn-uninitialized',
    'include_paths',
    'name',
]

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

CPP_OPTS = [
    'STAN_OPENCL',
    'OPENCL_DEVICE_ID',
    'OPENCL_PLATFORM_ID',
    'STAN_MPI',
    'STAN_THREADS',
]


class CompilerOptions:
    """
    User-specified flags for stanc and c++ compiler.

    Attributes:
        stanc_options - stanc compiler flags, options
        cpp_options - makefile options (NAME=value)
    """

    def __init__(
        self,
        stanc_options: Dict = None,
        cpp_options: Dict = None,
        logger: logging.Logger = None,
    ) -> None:
        """Initialize object."""
        self._stanc_options = stanc_options
        self._cpp_options = cpp_options
        self._logger = logger or get_logger()

    def __repr__(self) -> str:
        return 'stanc_options={}, cpp_options={}'.format(
            self._stanc_options, self._cpp_options
        )

    @property
    def stanc_options(self) -> Dict:
        """Stanc compiler options."""
        return self._stanc_options

    @property
    def cpp_options(self) -> Dict:
        """C++ compiler options."""
        return self._cpp_options

    def validate(self) -> None:
        """
        Check compiler args.
        Raise ValueError if invalid options are found.
        """
        self.validate_stanc_opts()
        self.validate_cpp_opts()

    def validate_stanc_opts(self) -> None:
        """
        Check stanc compiler args and consistency between stanc and c++ options.
        Raise ValueError if bad config is found.
        """
        # pylint: disable=no-member
        if self._stanc_options is None:
            return
        ignore = []
        paths = None
        for key, val in self._stanc_options.items():
            if key in STANC_IGNORE_OPTS:
                self._logger.info('ignoring compiler option: %s', key)
                ignore.append(key)
            elif key not in STANC_OPTS:
                raise ValueError(
                    'unknown stanc compiler option: {}'.format(key)
                )
            elif key == 'include_paths':
                paths = val
                if isinstance(val, str):
                    paths = val.split(',')
                elif not isinstance(val, list):
                    raise ValueError(
                        'Invalid include_paths, expecting list or '
                        'string, found type: {}.'.format(type(val))
                    )
            elif key == 'use-opencl':
                if self._cpp_options is None:
                    self._cpp_options = {'STAN_OPENCL': 'TRUE'}
                else:
                    self._cpp_options['STAN_OPENCL'] = 'TRUE'

        for opt in ignore:
            del self._stanc_options[opt]
        if paths is not None:
            self._stanc_options['include_paths'] = paths
            bad_paths = [
                dir
                for dir in self._stanc_options['include_paths']
                if not os.path.exists(dir)
            ]
            if any(bad_paths):
                raise ValueError(
                    'invalid include paths: {}'.format(', '.join(bad_paths))
                )

    def validate_cpp_opts(self) -> None:
        """
        Check cpp compiler args.
        Raise ValueError if bad config is found.
        """
        if self._cpp_options is None:
            return
        if (
            'OPENCL_DEVICE_ID' in self._cpp_options.keys()
            or 'OPENCL_PLATFORM_ID' in self._cpp_options.keys()
        ):
            self._cpp_options['STAN_OPENCL'] = 'TRUE'

        for key, val in self._cpp_options.items():
            if key not in CPP_OPTS:
                raise ValueError(
                    'unknown CmdStan makefile option: {}'.format(key)
                )
            if key in ['OPENCL_DEVICE_ID', 'OPENCL_PLATFORM_ID']:
                if not isinstance(val, int) or val < 0:
                    raise ValueError(
                        '{} must be a non-negative integer value,'
                        ' found {}.'.format(key, val)
                    )

    def add(self, new_opts: "CompilerOptions") -> None:  # noqa: disable=Q000
        """Adds options to existing set of compiler options."""
        if new_opts.stanc_options is not None:
            if self._stanc_options is None:
                self._stanc_options = new_opts.stanc_options
            else:
                for key, val in new_opts.stanc_options.items():
                    if key == 'include_paths':
                        self.add_include_path(str(val))
                    else:
                        self._stanc_options[key] = val
        if new_opts.cpp_options is not None:
            for key, val in new_opts.cpp_options.items():
                self._cpp_options[key] = val

    def add_include_path(self, path: str) -> None:
        """Adds include path to existing set of compiler options."""
        if 'include_paths' not in self._stanc_options:
            self._stanc_options['include_paths'] = [path]
        elif path not in self._stanc_options['include_paths']:
            self._stanc_options['include_paths'].append(path)

    def compose(self) -> List[str]:
        """Format makefile options as list of strings."""
        opts = []
        if self._stanc_options is not None and len(self._stanc_options) > 0:
            for key, val in self._stanc_options.items():
                if key == 'include_paths':
                    opts.append(
                        'STANCFLAGS+=--include_paths='
                        + ','.join(
                            (
                                Path(p).as_posix()
                                for p in self._stanc_options['include_paths']
                            )
                        )
                    )
                elif key == 'name':
                    opts.append('STANCFLAGS+=--{}={}'.format(key, val))
                else:
                    opts.append('STANCFLAGS+=--{}'.format(key))
        if self._cpp_options is not None and len(self._cpp_options) > 0:
            for key, val in self._cpp_options.items():
                opts.append('{}={}'.format(key, val))
        return opts
