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
   --verbose: flag, when specified prints output from CmdStan build process
   --progress: flag, when specified show progress bar for CmdStan download

   -c, --compiler : flag, add C++ compiler to path (Windows only)
"""
import argparse
import json
import os
import platform
import re
import subprocess
import sys
import tarfile
import urllib.error
import urllib.request
from collections import OrderedDict
from pathlib import Path
from time import sleep
from typing import Any, Callable, Dict, Optional

from cmdstanpy import _DOT_CMDSTAN, _DOT_CMDSTANPY
from cmdstanpy.utils import (
    cmdstan_path,
    get_logger,
    pushd,
    validate_dir,
    wrap_progress_hook,
)

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
        --verbose : show CmdStan build messages
        --progress : show progress bar for CmdStan download
        """

    if platform.system() == "Windows":
        msg += "-c (--compiler) : add C++ compiler to path (Windows only)\n"

    msg += "        -h (--help) : this message"

    print(msg)


def clean_all(verbose: bool = False) -> None:
    """
    Run `make clean-all` in the current directory (must be a cmdstan library).

    :param verbose: when ``True``, print build msgs to stdout.
    """
    cmd = [MAKE, 'clean-all']
    proc = subprocess.Popen(
        cmd,
        cwd=None,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=os.environ,
    )
    while proc.poll() is None:
        if proc.stdout:
            output = proc.stdout.readline().decode('utf-8').strip()
            if verbose and output:
                print(output, flush=True)
    _, stderr = proc.communicate()
    if proc.returncode:
        msgs = ['Command "make clean-all" failed']
        if stderr:
            msgs.append(stderr.decode('utf-8').strip())
        raise CmdStanInstallError('\n'.join(msgs))


def build(verbose: bool = False) -> None:
    """
    Run `make build` in the current directory (must be a cmdstan library)

    :param verbose: when ``True``, print build msgs to stdout.
    """
    cmd = [MAKE, 'build']
    proc = subprocess.Popen(
        cmd,
        cwd=None,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=os.environ,
    )
    while proc.poll() is None:
        if proc.stdout:
            output = proc.stdout.readline().decode('utf-8').strip()
            if verbose and output:
                print(output, flush=True)
    _, stderr = proc.communicate()
    if proc.returncode:
        msgs = ['Command "make build" failed']
        if stderr:
            msgs.append(stderr.decode('utf-8').strip())
        raise CmdStanInstallError('\n'.join(msgs))
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


def compile_example() -> None:
    """
    Compile the example model.
    The current directory must be a cmdstan library.
    """
    cmd = [
        MAKE,
        Path(
            os.path.join('examples', 'bernoulli', 'bernoulli' + EXTENSION)
        ).as_posix(),
    ]
    proc = subprocess.Popen(
        cmd,
        cwd=None,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=os.environ,
    )
    while proc.poll() is None:
        if proc.stdout:
            proc.stdout.readline().decode('utf-8')
    _, stderr = proc.communicate()
    if proc.returncode:
        msgs = ['Failed to compile example model bernoulli.stan']
        if stderr:
            msgs.append(stderr.decode('utf-8').strip())
        raise CmdStanInstallError('\n'.join(msgs))


def rebuild_cmdstan(verbose: bool = True) -> None:
    """
    Rebuilds the existing CmdStan installation.
    This assumes CmdStan has already been installed,
    though it need not be installed via CmdStanPy for
    this function to work.

    :param verbose:  Boolean value; when ``True``, output from CmdStan build
        processes will be streamed to the console.  Default is ``False``.
    """
    try:
        with pushd(cmdstan_path()):
            clean_all(verbose)
            build(verbose)
            compile_example()
    except ValueError as e:
        raise CmdStanInstallError(
            "Failed to rebuild CmdStan. Are you sure it is installed?"
        ) from e


def install_version(
    cmdstan_version: str, overwrite: bool = False, verbose: bool = False
) -> None:
    """
    Build specified CmdStan version by spawning subprocesses to
    run the Make utility on the downloaded CmdStan release src files.
    Assumes that current working directory is parent of release dir.

    :param cmdstan_version: CmdStan release, corresponds to release dirname.
    :param overwrite: when ``True``, run ``make clean-all`` before building.
    :param verbose: when ``True``, print build msgs to stdout.
    """
    with pushd(cmdstan_version):
        print('Building version {}'.format(cmdstan_version))
        if overwrite:
            print(
                'Overwrite requested, remove existing build of version '
                '{}'.format(cmdstan_version)
            )
            clean_all(verbose)
            print('Rebuilding version {}'.format(cmdstan_version))
        build(verbose)
        print('Test model compilation')
        compile_example()
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
            print('Release {} is unavailable from URL {}'.format(version, url))
            print('HTTPError: {}'.format(err.code))
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
            if progress:
                progress_hook: Optional[
                    Callable[[int, int, int], None]
                ] = wrap_progress_hook()
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
        tar = tarfile.open(file_tmp)
        target = os.getcwd()
        if platform.system() == 'Windows':
            # fixes long-path limitation on Windows
            target = r'\\?\{}'.format(target)
        tar.extractall(target)
    except Exception as e:  # pylint: disable=broad-except
        raise CmdStanInstallError(
            'Failed to unpack file {}'.format(file_tmp)
        ) from e
    finally:
        tar.close()
    print('Unpacked download as cmdstan-{}'.format(version))


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
    if not os.path.exists(cmdstan_dir):
        cmdstanpy_dir = os.path.expanduser(os.path.join('~', _DOT_CMDSTANPY))
        if os.path.exists(cmdstanpy_dir):
            cmdstan_dir = cmdstanpy_dir
            get_logger().warning(
                "Using ~/.cmdstanpy is deprecated and"
                " will not be automatically detected in version 1.0!\n"
                " Please rename to ~/.cmdstan"
            )

    install_dir = cmdstan_dir
    if args['dir']:
        install_dir = args['dir']

    validate_dir(install_dir)
    print('Install directory: {}'.format(install_dir))

    if args['progress']:
        progress = args['progress']
        try:
            # pylint: disable=unused-import
            from tqdm import tqdm  # noqa: F401
        except (ImportError, ModuleNotFoundError):
            progress = False
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

    cmdstan_version = 'cmdstan-{}'.format(version)
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
                )
            except RuntimeError as e:
                print(e)
                sys.exit(3)
        else:
            print('CmdStan version {} already installed'.format(version))


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


if __name__ == '__main__':
    main(parse_cmdline_args())
