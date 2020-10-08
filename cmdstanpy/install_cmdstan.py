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
   --overwrite: flag, when specified re-installs existing version.
   --verbose: flag, when specified prints output from CmdStan build process.

   -c, --compiler : add C++ compiler to path (Windows only)
"""
import argparse
import os
import platform
import subprocess
import sys
import tarfile
import urllib.request
import urllib.error
from pathlib import Path
from time import sleep

from cmdstanpy.utils import (
    get_dot_dir,
    get_latest_cmdstan,
    get_logger,
    pushd,
    validate_dir,
)

from cmdstanpy.install_cxx_toolchain import install_cxx_toolchain, has_cxx


EXTENSION = '.exe' if platform.system() == 'Windows' else ''


class CmdStanRetrieveError(RuntimeError):
    pass


class CmdStanInstallError(RuntimeError):
    pass


def usage():
    """Print usage."""
    print(
        """Arguments:
        -v (--version) :CmdStan version
        -d (--dir) : install directory
        --overwrite : replace installed version
        --verbose : show CmdStan build messages
        -h (--help) : this message
        """
    )


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
                if verbose:
                    print(proc.stdout.readline().decode('utf-8').strip())
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
            if verbose:
                print(proc.stdout.readline().decode('utf-8').strip())
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
        proc = subprocess.Popen(
            cmd,
            cwd=None,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=os.environ,
        )
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


def latest_version():
    """Report latest CmdStan release version."""
    for i in range(6):
        try:
            file_tmp, _ = urllib.request.urlretrieve(
                'https://api.github.com/repos/stan-dev/cmdstan/releases/latest'
            )
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
    with open(file_tmp, 'r') as fd:
        response = fd.read()
        start_idx = response.find('\"tag_name\":\"v') + len('"tag_name":"v')
        end_idx = response.find('\"', start_idx)
    return response[start_idx:end_idx]


def retrieve_version(version: str):
    """Download specified CmdStan version."""
    print('Downloading CmdStan version {}'.format(version))
    url = (
        'https://github.com/stan-dev/cmdstan/releases/download/'
        'v{0}/cmdstan-{0}.tar.gz'.format(version)
    )
    for i in range(6):  # always retry to allow for transient URLErrors
        try:
            file_tmp, _ = urllib.request.urlretrieve(url, filename=None)
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


def install_cmdstan(
    version: str = None,
    dir: str = None,
    overwrite: bool = False,
    verbose: bool = False,
) -> bool:
    """
    Download a CmdStan release from GitHub, then build executables and
    test model compilation. Default is to install latest CmdStan release.
    On successful install, sets the environment variable ``CMDSTAN``
    to the path of the installed release.

    Default behaviors can be overridden by specifying the release version
    and the base directory into which it is installed.
    An existing release version is only re-installed if argument ``overwrite``
    is  ``True``. By default ``overwrite`` is False.

    :param version: CmdStan version string, e.g. "2.24.1".
        Defaults to latest CmdStan release.

    :param dir: Path to the base directory into which all CmdStan versions
        are installed.  Defaults to hidden directory ``$HOME/.cmdstan`` (or
        ``$HOME/.cmdstanpy``, if the latter exists - backwards compatibility).
        If no directory is specified and neither of the above directories
        exist, directory ``$HOME/.cmdstan`` will be created and populated.

    :param overwrite:  Boolean value; when ``True``, will overwrite and
        rebuild an existing CmdStan installation.  Default is ``False``.

    :param verbose:  Boolean value; when ``True``, output from CmdStan build
        processes will be streamed to the console.  Default is ``False``.

    :return: Boolean value; ``True`` for success.
    """
    logger = get_logger()

    install_dir = dir
    if dir is None:
        install_dir = get_dot_dir()
    validate_dir(install_dir)
    print('Install directory: {}'.format(install_dir))

    if version is None:
        version = get_latest_cmdstan(install_dir)

    if is_version_available(version):
        print('Installing CmdStan version: {}'.format(version))
    else:
        raise ValueError(
            'Invalid version requested: {}, cannot install.'.format(version)
        )

    if platform.system() == 'Windows':
        # use compiler installed with install_cxx_toolchain
        # Install a new compiler if compiler not found
        # Search order is RTools40, RTools35
        from .utils import cxx_toolchain_path

        dot_dir = get_dot_dir()
        validate_dir(dot_dir)
        compiler_found = False
        for cxx_version in ['40', '35']:
            if has_cxx(dot_dir, cxx_version):
                compiler_found = True
                break
        if not compiler_found:
            logger.info('Installing RTools40')  # latest(?) RTools
            install_cxx_toolchain(version=cxx_version, dir=dot_dir)
            cxx_version = '40'
        # Add toolchain to $PATH
        cxx_toolchain_path(cxx_version)

    cmdstan_version = 'cmdstan-{}'.format(version)
    with pushd(install_dir):
        if overwrite or not (
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
            retrieve_version(version)
            install_version(
                cmdstan_version=cmdstan_version,
                overwrite=overwrite,
                verbose=verbose,
            )
        else:
            print('CmdStan version {} already installed'.format(version))

    # set path to newly installed version
    cmdstan_dir = dir
    if dir is None:
        cmdstan_dir = get_dot_dir()
    if version is None:
        version = get_latest_cmdstan(cmdstan_dir)
    new_cmdstan_path = os.path.join(cmdstan_dir, '-'.join(['cmdstan', version]))

    # validate, set path directly - avoid circular error messages
    stanc_bin = os.path.join(new_cmdstan_path, 'bin', 'stanc' + EXTENSION)
    if not os.path.exists(stanc_bin):
        logger.warning('CmdStan installation failed - missing stanc compiler')
        return False
    os.environ['CMDSTAN'] = new_cmdstan_path
    return True


if __name__ == '__main__':
    # pylint: disable=invalid-name
    parser = argparse.ArgumentParser()
    parser.add_argument('--version', '-v')
    parser.add_argument('--dir', '-d')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args(sys.argv[1:])

    try:
        if install_cmdstan(
            version=vars(args)['version'],
            dir=vars(args)['dir'],
            overwrite=vars(args)['overwrite'],
            verbose=vars(args)['verbose'],
        ):

            if vars(args)['version']:
                print(
                    'Installed CmdStan version {}.'.format(
                        vars(args)['version']
                    )
                )
            else:
                print('Installed latest CmdStan release.')
            print('Install directory: {}'.format(os.environ['CMDSTAN']))
            sys.exit(0)
        print("CmdStan install failed.")
        sys.exit(3)
    except RuntimeError as e:
        print("CmdStan install failed.")
        print(e)
        sys.exit(3)
