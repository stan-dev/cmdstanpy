#!/usr/bin/env python
"""
Download and install a C++ toolchain.
Currently implemented platforms (platform.system)
    Windows: RTools 3.5, 4.0 (default on x86), 4.4, 4.5 (default on ARM64)
    Darwin (macOS): Not implemented
    Linux: Not implemented
Optional command line arguments:
   -v, --version : version, defaults to latest
   -d, --dir : install directory, defaults to '~/.cmdstan
   -s (--silent) : install with /VERYSILENT instead of /SILENT for RTools
   -m --no-make : don't install mingw32-make (Windows RTools 4.0 only)
   --progress : flag, when specified show progress bar for RTools download
"""

import argparse
import os
import platform
import shutil
import subprocess
import sys
import urllib.request
from collections import OrderedDict
from time import sleep
from typing import Any

from cmdstanpy import _DOT_CMDSTAN
from cmdstanpy.utils import (
    determine_windows_arch,
    normalize_rtools_version,
    pushd,
    rtools_compiler,
    rtools_layouts,
    validate_dir,
    wrap_url_progress_hook,
)

EXTENSION = '.exe' if platform.system() == 'Windows' else ''

# CRAN embeds build revisions in the RTools 4.2+ installer filenames, so we
# use the r-hub mirror, which publishes them under a stable 'latest' tag.
# These are the builds the CmdStan guide points users at.
RTOOLS_INSTALLERS = {
    '4.5': {
        'x86_64': 'rtools45.exe',
        'aarch64': 'rtools45-aarch64.exe',
    },
    '4.4': {
        'x86_64': 'rtools44.exe',
        'aarch64': 'rtools44-aarch64.exe',
    },
}


def usage() -> None:
    """Print usage."""
    print(
        """Arguments:
        -v (--version) : RTools version: 3.5, 4.0, 4.4 or 4.5
        -d (--dir) : install directory
        -s (--silent) : install with /VERYSILENT instead of /SILENT for RTools
        -m (--no-make) : don't install mingw32-make (Windows RTools 4.0 only)
        --progress : flag, when specified show progress bar for RTools download
        -h (--help) : this message
        """
    )


def get_config(dir: str, silent: bool) -> list[str]:
    """Assemble config info."""
    config = []
    if platform.system() == 'Windows':
        _, dir = os.path.splitdrive(os.path.abspath(dir))
        if dir.startswith('\\'):
            dir = dir[1:]
        config = [
            '/SP-',
            '/VERYSILENT' if silent else '/SILENT',
            '/SUPPRESSMSGBOXES',
            '/CURRENTUSER',
            'LANG="English"',
            '/DIR="{}"'.format(dir),
            '/NOICONS',
            '/NORESTART',
        ]
    return config


def install_version(
    installation_dir: str,
    installation_file: str,
    version: str,
    silent: bool,
    verbose: bool = False,
) -> None:
    """Install specified toolchain version."""
    with pushd('.'):
        print(
            'Installing the C++ toolchain: {}'.format(
                os.path.splitext(installation_file)[0]
            )
        )
        cmd = [installation_file]
        cmd.extend(get_config(installation_dir, silent))
        print(' '.join(cmd))
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
                if output and verbose:
                    print(output, flush=True)
        _, stderr = proc.communicate()
        if proc.returncode:
            print('Installation failed: returncode={}'.format(proc.returncode))
            if stderr:
                print(stderr.decode('utf-8').strip())
            if is_installed(installation_dir, version):
                print('Installation files found at the installation location.')
            sys.exit(3)
    # check installation
    if is_installed(installation_dir, version):
        os.remove(installation_file)
    print('Installed {}'.format(os.path.splitext(installation_file)[0]))


def install_mingw32_make(toolchain_loc: str, verbose: bool = False) -> None:
    """Install mingw32-make for Windows RTools 4.0."""
    arch = determine_windows_arch()
    os.environ['PATH'] = ';'.join(
        list(
            OrderedDict.fromkeys(
                [
                    os.path.join(
                        toolchain_loc,
                        'mingw64' if arch == 'x86_64' else 'mingw32',
                        'bin',
                    ),
                    os.path.join(toolchain_loc, 'usr', 'bin'),
                ]
                + os.environ.get('PATH', '').split(';')
            )
        )
    )
    cmd = [
        'pacman',
        '-Sy',
        (
            'mingw-w64-x86_64-make'
            if arch == 'x86_64'
            else 'mingw-w64-i686-make'
        ),
        '--noconfirm',
    ]
    with pushd('.'):
        print(' '.join(cmd))
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
                if output and verbose:
                    print(output, flush=True)
        _, stderr = proc.communicate()
        if proc.returncode:
            print(
                'mingw32-make installation failed: returncode={}'.format(
                    proc.returncode
                )
            )
            if stderr:
                print(stderr.decode('utf-8').strip())
            sys.exit(3)
    print('Installed mingw32-make.exe')


def is_installed(toolchain_loc: str, version: str) -> bool:
    """Returns True is toolchain is installed."""
    if platform.system() != 'Windows':
        return False
    for layout in rtools_layouts(normalize_rtools_version(version)):
        tool_path = os.path.join(toolchain_loc, *layout.tool_subdir)
        if not os.path.exists(tool_path):
            continue
        if rtools_compiler(toolchain_loc, layout) is not None:
            return True
    return False


def latest_version() -> str:
    """Latest RTools version supported on this machine."""
    if platform.system() != 'Windows':
        return ''
    return '4.5'


def retrieve_toolchain(filename: str, url: str, progress: bool = True) -> None:
    """Download toolchain from URL."""
    print('Downloading C++ toolchain: {}'.format(filename))
    for i in range(6):
        try:
            if progress:
                progress_hook = wrap_url_progress_hook()
            else:
                progress_hook = None
            _ = urllib.request.urlretrieve(
                url, filename=filename, reporthook=progress_hook
            )
            break
        except urllib.error.URLError as err:
            print('Failed to download C++ toolchain')
            print(err)
            if i < 5:
                print('retry ({}/5)'.format(i + 1))
                sleep(1)
                continue
            sys.exit(3)
    print('Download successful, file: {}'.format(filename))


def normalize_version(version: str) -> str:
    """Return maj.min part of version string."""
    if platform.system() == 'Windows':
        return normalize_rtools_version(version)
    return version


def get_toolchain_name() -> str:
    """Return toolchain name."""
    if platform.system() == 'Windows':
        return 'RTools'
    return ''


# TODO(2.0): consider something other than RTools
def get_url(version: str, arch: str | None = None) -> str:
    """Return URL for toolchain."""
    if platform.system() != 'Windows':
        return ''
    if arch is None:
        arch = determine_windows_arch()
    if version in RTOOLS_INSTALLERS:
        installer = RTOOLS_INSTALLERS[version].get(arch, '')
        if not installer:
            return ''
        series = 'rtools' + version.replace('.', '')
        return (
            f'https://github.com/r-hub/{series}/releases/'
            f'download/latest/{installer}'
        )
    legacy = {
        ('4.0', 'x86_64'): 'rtools40-x86_64.exe',
        ('4.0', 'i686'): 'rtools40-i686.exe',
        ('3.5', 'x86_64'): 'Rtools35.exe',
        ('3.5', 'i686'): 'Rtools35.exe',
    }.get((version, arch), '')
    if legacy:
        return f'https://cran.r-project.org/bin/windows/Rtools/{legacy}'
    return ''


def get_toolchain_version(
    name: str, version: str, arch: str | None = None
) -> str:
    """Toolchain install folder name."""
    if platform.system() != 'Windows':
        return ''
    if arch is None:
        arch = determine_windows_arch()
    folder = '{}{}'.format(name, version.replace('.', ''))
    if arch == 'aarch64':
        folder += '-aarch64'
    return folder


def run_rtools_install(args: dict[str, Any]) -> None:
    """Main."""
    if platform.system() not in {'Windows'}:
        raise NotImplementedError(
            'Download for the C++ toolchain '
            'on the current platform has not '
            f'been implemented: {platform.system()}'
        )
    toolchain = get_toolchain_name()
    version = args['version']
    if version is None:
        version = latest_version()
    version = normalize_version(version)
    arch = determine_windows_arch()
    print(
        "C++ toolchain '{}' version: {} ({})".format(toolchain, version, arch)
    )

    url = get_url(version, arch)
    if not url:
        raise ValueError(
            f'RTools {version} is not available for {arch}. '
            f'Supported: {", ".join(sorted(RTOOLS_INSTALLERS))}, 4.0, 3.5 '
            '(4.0 and 3.5 are x86 only).'
        )

    if 'verbose' in args:
        verbose = args['verbose']
    else:
        verbose = False

    install_dir = args['dir']
    if install_dir is None:
        install_dir = os.path.expanduser(os.path.join('~', _DOT_CMDSTAN))
    validate_dir(install_dir)
    print('Install directory: {}'.format(install_dir))

    if 'progress' in args:
        progress = args['progress']
    else:
        progress = False

    if platform.system() == 'Windows':
        silent = 'silent' in args
        # force silent == False for 4.0 version
        if 'silent' not in args and version == '4.0':
            silent = False
    else:
        silent = False

    toolchain_folder = get_toolchain_version(toolchain, version, arch)
    with pushd(install_dir):
        if is_installed(toolchain_folder, version):
            print('C++ toolchain {} already installed'.format(toolchain_folder))
        else:
            if os.path.exists(toolchain_folder):
                shutil.rmtree(toolchain_folder, ignore_errors=False)
            retrieve_toolchain(
                toolchain_folder + EXTENSION, url, progress=progress
            )
            install_version(
                toolchain_folder,
                toolchain_folder + EXTENSION,
                version,
                silent,
                verbose,
            )
        if (
            'no-make' not in args
            and (platform.system() == 'Windows')
            and version == '4.0'
        ):
            if os.path.exists(
                os.path.join(
                    toolchain_folder, 'mingw64', 'bin', 'mingw32-make.exe'
                )
            ):
                print('mingw32-make.exe already installed')
            else:
                install_mingw32_make(toolchain_folder, verbose)


def parse_cmdline_args() -> dict[str, Any]:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--version',
        '-v',
        help="RTools version (3.5, 4.0, 4.4, 4.5), defaults to latest",
    )
    parser.add_argument(
        '--dir', '-d', help="install directory, defaults to '~/.cmdstan"
    )
    parser.add_argument(
        '--silent',
        '-s',
        action='store_true',
        help="install with /VERYSILENT instead of /SILENT for RTools",
    )
    parser.add_argument(
        '--no-make',
        '-m',
        action='store_false',
        help="don't install mingw32-make (Windows RTools 4.0 only)",
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help="flag, when specified prints output from RTools build process",
    )
    parser.add_argument(
        '--progress',
        action='store_true',
        help="flag, when specified show progress bar for CmdStan download",
    )
    return vars(parser.parse_args(sys.argv[1:]))


def __main__() -> None:
    run_rtools_install(parse_cmdline_args())


if __name__ == '__main__':
    __main__()
