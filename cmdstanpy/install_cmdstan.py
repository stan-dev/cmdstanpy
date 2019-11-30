#!/usr/bin/env python
"""
Download and install a CmdStan release from GitHub.
Optional command line arguments:
   -v, --version : version, defaults to latest
   -d, --dir : install directory, defaults to '~/.cmdstanpy
"""
import argparse
import contextlib
import os
import platform
import subprocess
import sys
import tarfile
import urllib.request
from pathlib import Path
from time import sleep

EXTENSION = '.exe' if platform.system() == 'Windows' else ''


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
        -h (--help) : this message
        """
    )


def install_version(cmdstan_version):
    """Build specified CmdStan version."""
    with pushd(cmdstan_version):
        print('Building {} binaries'.format(cmdstan_version))
        make = os.getenv(
            'MAKE', 'make' if platform.system() != 'Windows' else 'mingw32-make'
        )
        cmd = [make, 'build']
        proc = subprocess.Popen(
            cmd,
            cwd=None,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=os.environ,
        )
        _, stderr = proc.communicate()
        if proc.returncode:
            print('Command "make build" failed')
            if stderr:
                print(stderr.decode('utf-8').strip())
            sys.exit(3)
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
            print('Failed to compile example model bernoulli.stan')
            if stderr:
                print(stderr.decode('utf-8').strip())
            sys.exit(3)
    print('Installed {}'.format(cmdstan_version))


def is_installed(cmdstan_version):
    """Check if CmdStan is already installed."""
    if not os.path.exists(os.path.join(cmdstan_version, 'bin')):
        return False
    return os.path.exists(
        os.path.join(
            cmdstan_version, 'examples', 'bernoulli', 'bernoulli' + EXTENSION
        )
    )


def latest_version():
    """Report most recent CmdStan verion installed."""
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
            sys.exit(3)
    with open(file_tmp, 'r') as fd:
        response = fd.read()
        start_idx = response.find('\"tag_name\":\"v') + len('"tag_name":"v')
        end_idx = response.find('\"', start_idx)
    return response[start_idx:end_idx]


def retrieve_latest_version(version):
    """Download specified CmdStan version."""
    print('Downloading CmdStan version {}'.format(version))
    url = (
        'https://github.com/stan-dev/cmdstan/releases/download/'
        'v{0}/cmdstan-{0}.tar.gz'.format(version)
    )
    for i in range(6):
        try:
            file_tmp, _ = urllib.request.urlretrieve(url, filename=None)
            break
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
            sys.exit(3)
    print('Download successful, file: {}'.format(file_tmp))
    try:
        tar = tarfile.open(file_tmp)
        target = os.getcwd()
        if platform.system() == 'Windows':
            # fixes long-path limitation on Windows
            target = r'\\?\{}'.format(target)
        tar.extractall(target)
    except Exception as err:     # pylint: disable=broad-except
        print('Failed to unpack download')
        print(err)
        sys.exit(3)
    finally:
        tar.close()
    print('Unpacked download as cmdstan-{}'.format(version))


def validate_dir(install_dir):
    """Check that specified install directory exists, can write."""
    if not os.path.exists(install_dir):
        try:
            os.makedirs(install_dir)
        except OSError as e:
            raise ValueError(
                'Cannot create directory: {}'.format(install_dir)
            ) from e
    else:
        if not os.path.isdir(install_dir):
            raise ValueError(
                'File exists, should be a directory: {}'.format(install_dir)
            )
        try:
            with open('tmp_test_w', 'w') as fd:
                pass
            os.remove('tmp_test_w')  # cleanup
        except OSError as e:
            raise ValueError(
                'Cannot write files to directory {}'.format(install_dir)
            ) from e


def main():
    """Main."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--version', '-v')
    parser.add_argument('--dir', '-d')
    args = parser.parse_args(sys.argv[1:])

    version = vars(args)['version']
    if version is None:
        version = latest_version()
    print('CmdStan version: {}'.format(version))

    install_dir = vars(args)['dir']
    if install_dir is None:
        install_dir = os.path.expanduser(os.path.join('~', '.cmdstanpy'))
    validate_dir(install_dir)
    print('Install directory: {}'.format(install_dir))

    cmdstan_version = 'cmdstan-{}'.format(version)
    with pushd(install_dir):
        if not os.path.exists(cmdstan_version):
            retrieve_latest_version(version)
        if is_installed(cmdstan_version):
            print('CmdStan version {} already installed'.format(version))
        else:
            install_version(cmdstan_version)


if __name__ == '__main__':
    main()
