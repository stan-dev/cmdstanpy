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
import contextlib
import os
import platform
import subprocess
import sys
import tarfile
import urllib.request
import urllib.error
from pathlib import Path
from time import sleep

from cmdstanpy import _DOT_CMDSTAN, _DOT_CMDSTANPY

EXTENSION = '.exe' if platform.system() == 'Windows' else ''


class CmdStanRetrieveError(RuntimeError):
    pass


class CmdStanInstallError(RuntimeError):
    pass


@contextlib.contextmanager
def pushd(new_dir):
    """Acts like pushd/popd."""
    previous_dir = os.getcwd()
    os.chdir(new_dir)
    yield
    os.chdir(previous_dir)


def usage():
    """Print usage."""
    print(
        """Arguments:
        -v (--version) :CmdStan version
        -d (--dir) : install directory
        --overwrite : replace installed version
        -h (--help) : this message
        """
    )


def install_version(cmdstan_version, overwrite, verbose):
    """Build specified CmdStan version."""
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


def is_version_available(version):
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
        except urllib.error.URLError as err:
            if i < 5:
                print(
                    'checking version {} availability, retry ({}/5)'.format(
                        version, i + 1
                    )
                )
                sleep(1)
                continue
            print('Release {} is unavailable from URL {}'.format(version, url))
            print('URLError: {}'.format(err.reason))
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
        except urllib.error.URLError as err:
            print('Cannot connect to github.')
            print(err)
            if i < 5:
                print('retry ({}/5)'.format(i + 1))
                sleep(1)
                continue
            raise CmdStanRetrieveError(
                'Cannot connect to CmdStan github repo.'
            ) from err
    with open(file_tmp, 'r') as fd:
        response = fd.read()
        start_idx = response.find('\"tag_name\":\"v') + len('"tag_name":"v')
        end_idx = response.find('\"', start_idx)
    return response[start_idx:end_idx]


def retrieve_version(version):
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
        except urllib.error.HTTPError as err:
            raise CmdStanRetrieveError(
                'HTTPError: {}\n'
                'Version {} not available from github.com.'.format(
                    err.code, version
                )
            ) from err
        except urllib.error.URLError as err:
            print(
                'Failed to download CmdStan version {} from github.com'.format(
                    version
                )
            )
            print(err)
            if i < 5:
                print('retry ({}/5)'.format(i + 1))
                sleep(1)
                continue
            print('Version {} not available from github.com.'.format(version))
            raise CmdStanRetrieveError(
                'Version {} not available from github.com.'.format(version)
            ) from err
    print('Download successful, file: {}'.format(file_tmp))
    try:
        tar = tarfile.open(file_tmp)
        target = os.getcwd()
        if platform.system() == 'Windows':
            # fixes long-path limitation on Windows
            target = r'\\?\{}'.format(target)
        tar.extractall(target)
    except Exception as err:  # pylint: disable=broad-except
        raise CmdStanInstallError(
            'Failed to unpack file {}'.format(file_tmp)
        ) from err
    finally:
        tar.close()
    print('Unpacked download as cmdstan-{}'.format(version))


def validate_dir(install_dir):
    """Check that specified install directory exists, can write."""
    if not os.path.exists(install_dir):
        try:
            os.makedirs(install_dir)
        except (IOError, OSError, PermissionError) as err:
            raise ValueError(
                'Cannot create directory: {}'.format(install_dir)
            ) from err
    else:
        if not os.path.isdir(install_dir):
            raise ValueError(
                'File exists, should be a directory: {}'.format(install_dir)
            )
        try:
            with open('tmp_test_w', 'w'):
                pass
            os.remove('tmp_test_w')  # cleanup
        except (IOError, OSError, PermissionError) as err:
            raise ValueError(
                'Cannot write files to directory {}'.format(install_dir)
            ) from err


def main():
    """Main."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--version', '-v')
    parser.add_argument('--dir', '-d')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    if platform.system() == 'Windows':
        # use compiler installed with install_cxx_toolchain
        # Install a new compiler if compiler not found
        # Search order is RTools40, RTools35
        parser.add_argument(
            '--compiler', '-c', dest='compiler', action='store_true'
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
        cmdstan_dir = os.path.expanduser(os.path.join('~', _DOT_CMDSTANPY))

    install_dir = cmdstan_dir
    if vars(args)['dir']:
        install_dir = vars(args)['dir']

    validate_dir(install_dir)
    print('Install directory: {}'.format(install_dir))

    if platform.system() == 'Windows' and vars(args)['compiler']:
        from .install_cxx_toolchain import (
            main as _main_cxx,
            is_installed as _is_installed_cxx,
        )
        from .utils import cxx_toolchain_path

        cxx_loc = cmdstan_dir
        compiler_found = False
        for cxx_version in ['40', '35']:
            if _is_installed_cxx(cxx_loc, cxx_version):
                compiler_found = True
                break
        if not compiler_found:
            print('Installing RTools40')
            # copy argv and clear sys.argv
            original_argv = sys.argv[:]
            sys.argv = sys.argv[:1]
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
                retrieve_version(version)
                install_version(
                    cmdstan_version=cmdstan_version,
                    overwrite=vars(args)['overwrite'],
                    verbose=vars(args)['verbose'],
                )
            except RuntimeError as err:
                print(err)
                sys.exit(3)
        else:
            print('CmdStan version {} already installed'.format(version))


if __name__ == '__main__':
    main()
