#!/usr/bin/env python
"""
Download and install a CmdStan release from GitHub.
Downloads the release tar.gz file to temporary storage.
Retries GitHub requests in order to allow for transient network outages.
Builds CmdStan executables and tests the compiler by building
example model ``bernoulli.stan``.

Optional command line arguments:
   -v, --version <release> : version, defaults to latest release version
   -d, --dir <path> : install directory, defaults to '$HOME/.cmdstan
   --overwrite: flag, when specified re-installs existing version
   --progress: flag, when specified show progress bar for CmdStan download
   --verbose: flag, when specified prints output from CmdStan build process
   -c, --compiler : flag, add C++ compiler to path (Windows only)
"""
import argparse
import json
import os
import platform
import re
import sys
import tarfile
import urllib.error
import urllib.request
from collections import OrderedDict
from pathlib import Path
from time import sleep
from typing import Any, Callable, Dict, Optional

from tqdm.auto import tqdm

from cmdstanpy import _DOT_CMDSTAN
from cmdstanpy.utils import (
    cmdstan_path,
    do_command,
    pushd,
    validate_dir,
    wrap_url_progress_hook,
)

from . import progress as progbar

MAKE = os.getenv(
    'MAKE', 'make' if platform.system() != 'Windows' else 'mingw32-make'
)
EXTENSION = '.exe' if platform.system() == 'Windows' else ''


class CmdStanRetrieveError(RuntimeError):
    pass


class CmdStanInstallError(RuntimeError):
    pass


def usage() -> None:
    """Print usage."""
    msg = """
    Arguments:
        -v (--version) : CmdStan version
        -d (--dir) : install directory
        --overwrite : replace installed version
        --progress : show progress bar for CmdStan download
        --verbose : show outputs from installation processes
        """

    if platform.system() == "Windows":
        msg += "-c (--compiler) : add C++ compiler to path (Windows only)\n"

    msg += "        -h (--help) : this message"

    print(msg)


def clean_all(verbose: bool = False) -> None:
    """
    Run `make clean-all` in the current directory (must be a cmdstan library).

    :param verbose: Boolean value; when ``True``, show output from make command.
    """
    cmd = [MAKE, 'clean-all']
    try:
        if verbose:
            do_command(cmd)
        else:
            do_command(cmd, fd_out=None)

    except RuntimeError as e:
        # pylint: disable=raise-missing-from
        raise CmdStanInstallError(f'Command "make clean-all" failed\n{str(e)}')


def build(verbose: bool = False, progress: bool = True, cores: int = 1) -> None:
    """
    Run command ``make build`` in the current directory, which must be
    the home directory of a CmdStan version (or GitHub repo).
    By default, displays a progress bar which tracks make command outputs.
    If argument ``verbose=True``, instead of a progress bar, streams
    make command outputs to sys.stdout.  When both ``verbose`` and ``progress``
    are ``False``, runs silently.

    :param verbose: Boolean value; when ``True``, show output from make command.
        Default is ``False``.
    :param progress: Boolean value; when ``True`` display progress progress bar.
        Default is ``True``.
    :param cores: Integer, number of cores to use in the ``make`` command.
        Default is 1 core.
    """
    cmd = [MAKE, 'build', f'-j{cores}']
    try:
        if verbose:
            do_command(cmd)
        elif progress and progbar.allow_show_progress():
            progress_hook: Any = _wrap_build_progress_hook()
            do_command(cmd, fd_out=None, pbar=progress_hook)
        else:
            do_command(cmd, fd_out=None)

    except RuntimeError as e:
        # pylint: disable=raise-missing-from
        raise CmdStanInstallError(f'Command "make build" failed\n{str(e)}')
    if not os.path.exists(os.path.join('bin', 'stansummary' + EXTENSION)):
        raise CmdStanInstallError(
            f'bin/stansummary{EXTENSION} not found'
            ', please rebuild or report a bug!'
        )
    if not os.path.exists(os.path.join('bin', 'diagnose' + EXTENSION)):
        raise CmdStanInstallError(
            f'bin/stansummary{EXTENSION} not found'
            ', please rebuild or report a bug!'
        )

    if platform.system() == 'Windows':
        # Add tbb to the $PATH on Windows
        libtbb = os.path.join(
            os.getcwd(), 'stan', 'lib', 'stan_math', 'lib', 'tbb'
        )
        os.environ['PATH'] = ';'.join(
            list(
                OrderedDict.fromkeys(
                    [libtbb] + os.environ.get('PATH', '').split(';')
                )
            )
        )


@progbar.wrap_callback
def _wrap_build_progress_hook() -> Optional[Callable[[str], None]]:
    """Sets up tqdm callback for CmdStan sampler console msgs."""
    pad = ' ' * 20
    msgs_expected = 150  # hack: 2.27 make build send ~140 msgs to console
    pbar: tqdm = tqdm(
        total=msgs_expected,
        bar_format="{desc} ({elapsed}) | {bar} | {postfix[0][value]}",
        postfix=[dict(value=f'Building CmdStan {pad}')],
        colour='blue',
        desc='',
        position=0,
    )

    def build_progress_hook(line: str) -> None:
        if line.startswith('--- CmdStan'):
            pbar.set_description('Done')
            pbar.postfix[0]["value"] = line
            pbar.update(msgs_expected - pbar.n)
            pbar.close()
        else:
            if line.startswith('--'):
                pbar.postfix[0]["value"] = line
            else:
                pbar.postfix[0]["value"] = f'{line[:8]} ... {line[-20:]}'
                pbar.set_description('Compiling')
                pbar.update(1)

    return build_progress_hook


def compile_example(verbose: bool = False) -> None:
    """
    Compile the example model.
    The current directory must be a cmdstan installation, i.e.,
    contains the makefile, Stanc compiler, and all libraries.

    :param verbose: Boolean value; when ``True``, show output from make command.
    """
    cmd = [
        MAKE,
        Path(
            os.path.join('examples', 'bernoulli', 'bernoulli' + EXTENSION)
        ).as_posix(),
    ]
    try:
        if verbose:
            do_command(cmd)
        else:
            do_command(cmd, fd_out=None)
    except RuntimeError as e:
        # pylint: disable=raise-missing-from
        raise CmdStanInstallError(f'Command "make clean-all" failed\n{e}')


def rebuild_cmdstan(
    verbose: bool = False, progress: bool = True, cores: int = 1
) -> None:
    """
    Rebuilds the existing CmdStan installation.
    This assumes CmdStan has already been installed,
    though it need not be installed via CmdStanPy for
    this function to work.

    :param verbose: Boolean value; when ``True``, show output from make command.
        Default is ``False``.
    :param progress: Boolean value; when ``True`` display progress progress bar.
        Default is ``True``.
    :param cores: Integer, number of cores to use in the ``make`` command.
        Default is 1 core.
    """
    try:
        with pushd(cmdstan_path()):
            clean_all(verbose)
            build(verbose, progress, cores)
            compile_example(verbose)
    except ValueError as e:
        raise CmdStanInstallError(
            "Failed to rebuild CmdStan. Are you sure it is installed?"
        ) from e


def install_version(
    cmdstan_version: str,
    overwrite: bool = False,
    verbose: bool = False,
    progress: bool = True,
    cores: int = 1,
) -> None:
    """
    Build specified CmdStan version by spawning subprocesses to
    run the Make utility on the downloaded CmdStan release src files.
    Assumes that current working directory is parent of release dir.

    :param cmdstan_version: CmdStan release, corresponds to release dirname.
    :param overwrite: when ``True``, run ``make clean-all`` before building.
    :param verbose: Boolean value; when ``True``, show output from make command.
    """
    with pushd(cmdstan_version):
        print(
            'Building version {}, may take several minutes, '
            'depending on your system.'.format(cmdstan_version)
        )
        if overwrite:
            print(
                'Overwrite requested, remove existing build of version '
                '{}'.format(cmdstan_version)
            )
            clean_all(verbose)
            print('Rebuilding version {}'.format(cmdstan_version))
        build(verbose, progress=progress, cores=cores)
        print('Test model compilation')
        compile_example(verbose)
    print('Installed {}'.format(cmdstan_version))


def is_version_available(version: str) -> bool:
    is_available = True
    url = (
        'https://github.com/stan-dev/cmdstan/releases/download/'
        'v{0}/cmdstan-{0}.tar.gz'.format(version)
    )
    for i in range(6):
        try:
            urllib.request.urlopen(url)
        except urllib.error.HTTPError as err:
            print(f'Release {version} is unavailable from URL {url}')
            print(f'HTTPError: {err.code}')
            is_available = False
            break
        except urllib.error.URLError as e:
            if i < 5:
                print(
                    'checking version {} availability, retry ({}/5)'.format(
                        version, i + 1
                    )
                )
                sleep(1)
                continue
            print('Release {} is unavailable from URL {}'.format(version, url))
            print('URLError: {}'.format(e.reason))
            is_available = False
    return is_available


def get_headers() -> Dict[str, str]:
    """Create headers dictionary."""
    headers = {}
    GITHUB_PAT = os.environ.get("GITHUB_PAT")  # pylint:disable=invalid-name
    if GITHUB_PAT is not None:
        headers["Authorization"] = "token {}".format(GITHUB_PAT)
    return headers


def latest_version() -> str:
    """Report latest CmdStan release version."""
    url = 'https://api.github.com/repos/stan-dev/cmdstan/releases/latest'
    request = urllib.request.Request(url, headers=get_headers())
    for i in range(6):
        try:
            response = urllib.request.urlopen(request).read()
            break
        except urllib.error.URLError as e:
            print('Cannot connect to github.')
            print(e)
            if i < 5:
                print('retry ({}/5)'.format(i + 1))
                sleep(1)
                continue
            raise CmdStanRetrieveError(
                'Cannot connect to CmdStan github repo.'
            ) from e
    content = json.loads(response.decode('utf-8'))
    tag = content['tag_name']
    match = re.search(r'v?(.+)', tag)
    if match is not None:
        tag = match.group(1)
    return tag  # type: ignore


def retrieve_version(version: str, progress: bool = True) -> None:
    """Download specified CmdStan version."""
    if version is None or version == '':
        raise ValueError('Argument "version" unspecified.')
    print('Downloading CmdStan version {}'.format(version))
    url = (
        'https://github.com/stan-dev/cmdstan/releases/download/'
        'v{0}/cmdstan-{0}.tar.gz'.format(version)
    )
    for i in range(6):  # always retry to allow for transient URLErrors
        try:
            if progress and progbar.allow_show_progress():
                progress_hook: Optional[
                    Callable[[int, int, int], None]
                ] = wrap_url_progress_hook()
            else:
                progress_hook = None
            file_tmp, _ = urllib.request.urlretrieve(
                url, filename=None, reporthook=progress_hook
            )
            break
        except urllib.error.HTTPError as e:
            raise CmdStanRetrieveError(
                'HTTPError: {}\n'
                'Version {} not available from github.com.'.format(
                    e.code, version
                )
            ) from e
        except urllib.error.URLError as e:
            print(
                'Failed to download CmdStan version {} from github.com'.format(
                    version
                )
            )
            print(e)
            if i < 5:
                print('retry ({}/5)'.format(i + 1))
                sleep(1)
                continue
            print('Version {} not available from github.com.'.format(version))
            raise CmdStanRetrieveError(
                'Version {} not available from github.com.'.format(version)
            ) from e
    print('Download successful, file: {}'.format(file_tmp))
    try:
        print('Extracting distribution')
        tar = tarfile.open(file_tmp)
        first = tar.next()
        if first is not None:
            top_dir = first.name
        cmdstan_dir = f'cmdstan-{version}'
        if top_dir != cmdstan_dir:
            raise CmdStanInstallError(
                'tarfile should contain top-level dir {},'
                'but found dir {} instead.'.format(cmdstan_dir, top_dir)
            )
        target = os.getcwd()
        if platform.system() == 'Windows':
            # fixes long-path limitation on Windows
            target = r'\\?\{}'.format(target)

        if progress and progbar.allow_show_progress():
            for member in tqdm(
                iterable=tar.getmembers(),
                total=len(tar.getmembers()),
                colour='blue',
                leave=False,
            ):
                tar.extract(member=member)
        else:
            tar.extractall()
    except Exception as e:  # pylint: disable=broad-except
        raise CmdStanInstallError(
            f'Failed to unpack file {file_tmp}, error:\n\t{str(e)}'
        ) from e
    finally:
        tar.close()
    print(f'Unpacked download as {cmdstan_dir}')


def parse_cmdline_args() -> Dict[str, Any]:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--version', '-v', help="version, defaults to latest release version"
    )
    parser.add_argument(
        '--dir', '-d', help="install directory, defaults to '$HOME/.cmdstan"
    )
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help="flag, when specified re-installs existing version",
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help="flag, when specified prints output from CmdStan build process",
    )
    parser.add_argument(
        '--progress',
        action='store_true',
        help="flag, when specified show progress bar for CmdStan download",
    )
    parser.add_argument(
        "--cores",
        default=1,
        type=int,
        help="number of cores to use while building",
    )
    if platform.system() == 'Windows':
        # use compiler installed with install_cxx_toolchain
        # Install a new compiler if compiler not found
        # Search order is RTools40, RTools35
        parser.add_argument(
            '--compiler',
            '-c',
            dest='compiler',
            action='store_true',
            help="flag, add C++ compiler to path (Windows only)",
        )
    return vars(parser.parse_args(sys.argv[1:]))


def main(args: Dict[str, Any]) -> None:
    """Main."""

    version = latest_version()
    if args['version']:
        version = args['version']

    if is_version_available(version):
        print('Installing CmdStan version: {}'.format(version))
    else:
        raise ValueError(
            'Invalid version requested: {}, cannot install.'.format(version)
        )

    cmdstan_dir = os.path.expanduser(os.path.join('~', _DOT_CMDSTAN))

    install_dir = cmdstan_dir
    if args['dir']:
        install_dir = args['dir']

    validate_dir(install_dir)
    print('Install directory: {}'.format(install_dir))

    if args['progress']:
        progress = args['progress']
    else:
        progress = False

    if platform.system() == 'Windows' and args['compiler']:
        from .install_cxx_toolchain import is_installed as _is_installed_cxx
        from .install_cxx_toolchain import main as _main_cxx
        from .utils import cxx_toolchain_path

        cxx_loc = cmdstan_dir
        compiler_found = False
        rtools40_home = os.environ.get('RTOOLS40_HOME')
        for cxx_loc in (
            [rtools40_home] if rtools40_home is not None else []
        ) + [
            cmdstan_dir,
            os.path.join(os.path.abspath("/"), "RTools40"),
            os.path.join(os.path.abspath("/"), "RTools"),
            os.path.join(os.path.abspath("/"), "RTools35"),
            os.path.join(os.path.abspath("/"), "RBuildTools"),
        ]:
            for cxx_version in ['40', '35']:
                if _is_installed_cxx(cxx_loc, cxx_version):
                    compiler_found = True
                    break
            if compiler_found:
                break
        if not compiler_found:
            print('Installing RTools40')
            # copy argv and clear sys.argv
            cxx_args = {k: v for k, v in args.items() if k != 'compiler'}
            _main_cxx(cxx_args)
            cxx_version = '40'
        # Add toolchain to $PATH
        cxx_toolchain_path(cxx_version, args['dir'])

    cmdstan_version = f'cmdstan-{version}'
    with pushd(install_dir):
        if args['overwrite'] or not (
            os.path.exists(cmdstan_version)
            and os.path.exists(
                os.path.join(
                    cmdstan_version,
                    'examples',
                    'bernoulli',
                    'bernoulli' + EXTENSION,
                )
            )
        ):
            try:
                retrieve_version(version, progress)
                install_version(
                    cmdstan_version=cmdstan_version,
                    overwrite=args['overwrite'],
                    verbose=args['verbose'],
                    progress=progress,
                    cores=args['cores'],
                )
            except RuntimeError as e:
                print(e)
                sys.exit(3)
        else:
            print('CmdStan version {} already installed'.format(version))


def __main__() -> None:
    main(parse_cmdline_args())


if __name__ == '__main__':
    __main__()
