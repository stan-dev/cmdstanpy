"""
Utilities for interacting with the filesystem on multiple platforms
"""
import contextlib
import os
import platform
import shutil
import tempfile
from typing import Any, Iterator, List, Mapping, Tuple, Union

from cmdstanpy import _TMPDIR

from .json import write_stan_json
from .logging import get_logger

EXTENSION = '.exe' if platform.system() == 'Windows' else ''


def windows_short_path(path: str) -> str:
    """
    Gets the short path name of a given long path.
    http://stackoverflow.com/a/23598461/200291

    On non-Windows platforms, returns the path

    If (base)path does not exist, function raises RuntimeError
    """
    if platform.system() != 'Windows':
        return path

    if os.path.isfile(path) or (
        not os.path.isdir(path) and os.path.splitext(path)[1] != ''
    ):
        base_path, file_name = os.path.split(path)
    else:
        base_path, file_name = path, ''

    if not os.path.exists(base_path):
        raise RuntimeError(
            'Windows short path function needs a valid directory. '
            'Base directory does not exist: "{}"'.format(base_path)
        )

    import ctypes
    from ctypes import wintypes

    # pylint: disable=invalid-name
    _GetShortPathNameW = (
        ctypes.windll.kernel32.GetShortPathNameW  # type: ignore
    )

    _GetShortPathNameW.argtypes = [
        wintypes.LPCWSTR,
        wintypes.LPWSTR,
        wintypes.DWORD,
    ]
    _GetShortPathNameW.restype = wintypes.DWORD

    output_buf_size = 0
    while True:
        output_buf = ctypes.create_unicode_buffer(output_buf_size)
        needed = _GetShortPathNameW(base_path, output_buf, output_buf_size)
        if output_buf_size >= needed:
            short_base_path = output_buf.value
            break
        else:
            output_buf_size = needed

    short_path = (
        os.path.join(short_base_path, file_name)
        if file_name
        else short_base_path
    )
    return short_path


def create_named_text_file(
    dir: str, prefix: str, suffix: str, name_only: bool = False
) -> str:
    """
    Create a named unique file, return filename.
    Flag 'name_only' will create then delete the tmp file;
    this lets us create filename args for commands which
    disallow overwriting existing files (e.g., 'stansummary').
    """
    fd = tempfile.NamedTemporaryFile(
        mode='w+', prefix=prefix, suffix=suffix, dir=dir, delete=name_only
    )
    path = fd.name
    fd.close()
    return path


@contextlib.contextmanager
def pushd(new_dir: str) -> Iterator[None]:
    """Acts like pushd/popd."""
    previous_dir = os.getcwd()
    os.chdir(new_dir)
    try:
        yield
    finally:
        os.chdir(previous_dir)


class MaybeDictToFilePath:
    """Context manager for json files."""

    def __init__(
        self,
        *objs: Union[
            str, Mapping[str, Any], List[Any], int, float, os.PathLike, None
        ],
    ):
        self._unlink = [False] * len(objs)
        self._paths: List[Any] = [''] * len(objs)
        i = 0
        # pylint: disable=isinstance-second-argument-not-valid-type
        for obj in objs:
            if isinstance(obj, Mapping):
                data_file = create_named_text_file(
                    dir=_TMPDIR, prefix='', suffix='.json'
                )
                get_logger().debug('input tempfile: %s', data_file)
                write_stan_json(data_file, obj)
                self._paths[i] = data_file
                self._unlink[i] = True
            elif isinstance(obj, (str, os.PathLike)):
                if not os.path.exists(obj):
                    raise ValueError("File doesn't exist {}".format(obj))
                self._paths[i] = obj
            elif isinstance(obj, list):
                err_msgs = []
                missing_obj_items = []
                for j, obj_item in enumerate(obj):
                    if not isinstance(obj_item, str):
                        err_msgs.append(
                            (
                                'List element {} must be a filename string,'
                                ' found {}'
                            ).format(j, obj_item)
                        )
                    elif not os.path.exists(obj_item):
                        missing_obj_items.append(
                            "File doesn't exist: {}".format(obj_item)
                        )
                if err_msgs:
                    raise ValueError('\n'.join(err_msgs))
                if missing_obj_items:
                    raise ValueError('\n'.join(missing_obj_items))
                self._paths[i] = obj
            elif obj is None:
                self._paths[i] = None
            elif i == 1 and isinstance(obj, (int, float)):
                self._paths[i] = obj
            else:
                raise ValueError('data must be string or dict')
            i += 1

    def __enter__(self) -> List[str]:
        return self._paths

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:  # type: ignore
        for can_unlink, path in zip(self._unlink, self._paths):
            if can_unlink and path:
                try:
                    os.remove(path)
                except PermissionError:
                    pass


class SanitizedOrTmpFilePath:
    """Context manager for tmpfiles, handles spaces in filepath."""

    def __init__(self, file_path: str):
        self._tmpdir = None
        if ' ' in os.path.abspath(file_path) and platform.system() == 'Windows':
            base_path, file_name = os.path.split(os.path.abspath(file_path))
            os.makedirs(base_path, exist_ok=True)
            try:
                short_base_path = windows_short_path(base_path)
                if os.path.exists(short_base_path):
                    file_path = os.path.join(short_base_path, file_name)
            except RuntimeError:
                pass

        if ' ' in os.path.abspath(file_path):
            tmpdir = tempfile.mkdtemp()
            if ' ' in tmpdir:
                raise RuntimeError(
                    'Unable to generate temporary path without spaces! \n'
                    + 'Please move your stan file to location without spaces.'
                )

            _, path = tempfile.mkstemp(suffix='.stan', dir=tmpdir)

            shutil.copy(file_path, path)
            self._path = path
            self._tmpdir = tmpdir
        else:
            self._path = file_path

    def __enter__(self) -> Tuple[str, bool]:
        return self._path, self._tmpdir is not None

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:  # type: ignore
        if self._tmpdir:
            shutil.rmtree(self._tmpdir, ignore_errors=True)
