#!/usr/bin/env python
"""
Download and install a CmdStan release from GitHub.
Downloads the release tar.gz file to temporary storage.
Retries GitHub requests in order to allow for transient network outages.
Builds CmdStan executables and tests the compiler by building
example model ``bernoulli.stan``.

Optional command line arguments:
   -v, --version <release> : version, defaults to latest release version
   -d, --dir <path> : install directory, defaults to '$HOME/.cmdstan(py)
   --overwrite: flag, when specified re-installs existing version
   --verbose: flag, when specified prints output from CmdStan build process
   --progress: flag, when specified show progress bar for CmdStan download

   -c, --compiler : flag, add C++ compiler to path (Windows only)
"""
import argparse
import contextlib
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

from cmdstanpy import _DOT_CMDSTAN, _DOT_CMDSTANPY
from cmdstanpy.utils import validate_dir

EXTENSION = '.exe' if platform.system() == 'Windows' else ''


class CmdStanRetrieveError(RuntimeError):
    pass


class CmdStanInstallError(RuntimeError):
    pass


@contextlib.contextmanager
def pushd(new_dir: str):
    """Acts like pushd/popd."""
    previous_dir = os.getcwd()
    os.chdir(new_dir)
    yield
    os.chdir(previous_dir)


def usage():
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


def install_version(
    cmdstan_version: str, overwrite: bool = False, verbose: bool = False
):
    """
    Build specified CmdStan version by spawning subprocesses to
    run the Make utility on the downloaded CmdStan release src files.
    Assumes that current working directory is parent of release dir.

    :param cmdstan_version: CmdStan release, corresponds to release dirname.
    :param overwrite: when ``True``, run ``make clean-all`` before building.
    :param verbose: when ``True``, print build msgs to stdout.
    """
    with pushd(cmdstan_version):
        make = os.getenv(
            'MAKE', 'make' if platform.system() != 'Windows' else 'mingw32-make'
        )
        print('Building version {}'.format(cmdstan_version))
        if overwrite:
            print(
                'Overwrite requested, remove existing build of version '
                '{}'.format(cmdstan_version)
            )
            cmd = [make, 'clean-all']
            proc = subprocess.Popen(
                cmd,
                cwd=None,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=os.environ,
            )
            while proc.poll() is None:
                output = proc.stdout.readline().decode('utf-8').strip()
                if verbose and output:
                    print(output, flush=True)
            _, stderr = proc.communicate()
            if proc.returncode:
                msgs = ['Command "make clean-all" failed']
                if stderr:
                    msgs.append(stderr.decode('utf-8').strip())
                raise CmdStanInstallError('\n'.join(msgs))
            print('Rebuilding version {}'.format(cmdstan_version))
        cmd = [make, 'build']
        proc = subprocess.Popen(
            cmd,
            cwd=None,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=os.environ,
        )
        while proc.poll() is None:
            output = proc.stdout.readline().decode('utf-8').strip()
            if verbose and output:
                print(output, flush=True)
        _, stderr = proc.communicate()
        if proc.returncode:
            msgs = ['Command "make build" failed']
            if stderr:
                msgs.append(stderr.decode('utf-8').strip())
            raise CmdStanInstallError('\n'.join(msgs))
        print('Test model compilation')
        cmd = [
            make,
            Path(
                os.path.join('examples', 'bernoulli', 'bernoulli' + EXTENSION)
            ).as_posix(),
        ]
        if platform.system() == "Windows":
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
        proc = subprocess.Popen(
            cmd,
            cwd=None,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=os.environ,
        )
        while proc.poll() is None:
            proc.stdout.readline().decode('utf-8')
        _, stderr = proc.communicate()
        if proc.returncode:
            msgs = ['Failed to compile example model bernoulli.stan']
            if stderr:
                msgs.append(stderr.decode('utf-8').strip())
            raise CmdStanInstallError('\n'.join(msgs))
    print('Installed {}'.format(cmdstan_version))


def is_version_available(version: str):
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


def get_headers():
    """Create headers dictionary."""
    headers = {}
    GITHUB_PAT = os.environ.get("GITHUB_PAT")  # pylint:disable=invalid-name
    if GITHUB_PAT is not None:
        headers["Authorization"] = "token {}".format(GITHUB_PAT)
    return headers


def latest_version():
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
    return tag


def wrap_progress_hook():
    try:
        from tqdm import tqdm

        pbar = tqdm(
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        )

        def download_progress_hook(count, block_size, total_size):
            if pbar.total is None:
                pbar.total = total_size
                pbar.reset()
            downloaded_size = count * block_size
            pbar.update(downloaded_size - pbar.n)
            if pbar.n >= total_size:
                pbar.close()

    except (ImportError, ModuleNotFoundError):
        print("tqdm is not installed, progressbar not shown")
        download_progress_hook = None

    return download_progress_hook


def retrieve_version(version: str, progress=True):
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
                progress_hook = wrap_progress_hook()
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


def main():
    """Main."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--version', '-v', help="version, defaults to latest release version"
    )
    parser.add_argument(
        '--dir', '-d', help="install directory, defaults to '$HOME/.cmdstan(py)"
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
    args = parser.parse_args(sys.argv[1:])

    version = latest_version()
    if vars(args)['version']:
        version = vars(args)['version']

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

    install_dir = cmdstan_dir
    if vars(args)['dir']:
        install_dir = vars(args)['dir']

    validate_dir(install_dir)
    print('Install directory: {}'.format(install_dir))

    if vars(args)['progress']:
        progress = vars(args)['progress']
        try:
            # pylint: disable=unused-import
            from tqdm import tqdm  # noqa: F401
        except (ImportError, ModuleNotFoundError):
            progress = False
    else:
        progress = False

    if platform.system() == 'Windows' and vars(args)['compiler']:
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
            original_argv = sys.argv[:]
            sys.argv = [
                item for item in sys.argv if item not in ("--compiler", "-c")
            ]
            _main_cxx()
            sys.argv = original_argv
            cxx_version = '40'
        # Add toolchain to $PATH
        cxx_toolchain_path(cxx_version)

    cmdstan_version = 'cmdstan-{}'.format(version)
    with pushd(install_dir):
        if vars(args)['overwrite'] or not (
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
                    overwrite=vars(args)['overwrite'],
                    verbose=vars(args)['verbose'],
                )
            except RuntimeError as e:
                print(e)
                sys.exit(3)
        else:
            print('CmdStan version {} already installed'.format(version))


if __name__ == '__main__':
    main()
