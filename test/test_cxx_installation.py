"""install_cxx_toolchain tests"""

# pylint: disable=redefined-outer-name

import os
import platform
import sys
from pathlib import Path
from test import mark_not_windows, mark_windows_only
from typing import Callable

import pytest

from cmdstanpy import install_cxx_toolchain
from cmdstanpy.utils import cxx_toolchain_path

SetArch = Callable[[str], None]

# (rtools version, arch) -> (compiler subdir, tool subdir, compiler exe)
LAYOUTS = {
    ('3.5', 'x86_64'): (('mingw_64', 'bin'), ('bin',), 'g++'),
    ('4.0', 'x86_64'): (('mingw64', 'bin'), ('usr', 'bin'), 'g++'),
    ('4.4', 'x86_64'): (
        ('x86_64-w64-mingw32.static.posix', 'bin'),
        ('usr', 'bin'),
        'g++',
    ),
    ('4.5', 'x86_64'): (
        ('x86_64-w64-mingw32.static.posix', 'bin'),
        ('usr', 'bin'),
        'g++',
    ),
    ('4.4', 'aarch64'): (
        ('aarch64-w64-mingw32.static.posix', 'bin'),
        ('usr', 'bin'),
        'clang++',
    ),
    ('4.5', 'aarch64'): (
        ('aarch64-w64-mingw32.static.posix', 'bin'),
        ('usr', 'bin'),
        'clang++',
    ),
}

MACHINES = {'x86_64': 'AMD64', 'aarch64': 'ARM64', 'i686': 'x86'}


@pytest.fixture(autouse=True)
def clean_env(
    monkeypatch: pytest.MonkeyPatch, tmp_path_factory: pytest.TempPathFactory
) -> None:
    """Hide any RTools installation the machine already has."""
    for var in (
        'CMDSTAN_TOOLCHAIN',
        'RTOOLS45_HOME',
        'RTOOLS44_HOME',
        'RTOOLS43_HOME',
        'RTOOLS42_HOME',
        'RTOOLS40_HOME',
    ):
        monkeypatch.delenv(var, raising=False)
    # cxx_toolchain_path mutates PATH in place; let monkeypatch restore it
    monkeypatch.setenv('PATH', os.environ.get('PATH', ''))
    home = tmp_path_factory.mktemp('home')
    monkeypatch.setenv('HOME', str(home))
    monkeypatch.setenv('USERPROFILE', str(home))


@pytest.fixture
def set_arch(monkeypatch: pytest.MonkeyPatch) -> SetArch:
    """Pretend to be running on a given architecture."""

    def _set(arch: str) -> None:
        monkeypatch.setattr(platform, 'machine', lambda: MACHINES[arch])

    return _set


def make_toolchain(
    root: Path, version: str, arch: str, complete: bool = True
) -> str:
    """Create a fake RTools installation tree, return its root."""
    compiler_subdir, tool_subdir, compiler = LAYOUTS[(version, arch)]
    compiler_dir = root.joinpath(*compiler_subdir)
    compiler_dir.mkdir(parents=True)
    if complete:
        compiler_dir.joinpath(compiler + '.exe').write_text('')
        root.joinpath(*tool_subdir).mkdir(parents=True, exist_ok=True)
    return str(root)


# ---------------------------------------------------------------------------
# version / url resolution
# ---------------------------------------------------------------------------


@mark_windows_only
@pytest.mark.parametrize(
    'given,expected',
    [
        ('4.0', '4.0'),
        ('4', '4.0'),
        ('40', '4.0'),
        ('3.5', '3.5'),
        ('35', '3.5'),
        ('3', '3.5'),
        ('4.2', '4.2'),
        ('42', '4.2'),
        ('4.4', '4.4'),
        ('44', '4.4'),
        ('4.5', '4.5'),
        ('45', '4.5'),
    ],
)
def test_normalize_version(given: str, expected: str) -> None:
    assert install_cxx_toolchain.normalize_version(given) == expected


@mark_windows_only
def test_toolchain_name() -> None:
    assert install_cxx_toolchain.get_toolchain_name() == 'RTools'


@mark_windows_only
@pytest.mark.parametrize(
    'version,arch,expected',
    [
        (
            '4.5',
            'aarch64',
            'https://cran.r-project.org/bin/windows/Rtools/rtools45/files/'
            'rtools45-aarch64-6768-6492.exe',
        ),
        (
            '4.5',
            'x86_64',
            'https://cran.r-project.org/bin/windows/Rtools/rtools45/files/'
            'rtools45-6768-6492.exe',
        ),
        (
            '4.4',
            'aarch64',
            'https://cran.r-project.org/bin/windows/Rtools/rtools44/files/'
            'rtools44-aarch64-6459-6401.exe',
        ),
        (
            '4.4',
            'x86_64',
            'https://cran.r-project.org/bin/windows/Rtools/rtools44/files/'
            'rtools44-6459-6401.exe',
        ),
        (
            '4.0',
            'x86_64',
            'https://cran.r-project.org/bin/windows/Rtools/'
            'rtools40-x86_64.exe',
        ),
        (
            '3.5',
            'x86_64',
            'https://cran.r-project.org/bin/windows/Rtools/Rtools35.exe',
        ),
        # no ARM64 builds exist before RTools 4.4
        ('4.0', 'aarch64', ''),
        ('3.5', 'aarch64', ''),
    ],
)
def test_get_url(version: str, arch: str, expected: str) -> None:
    assert install_cxx_toolchain.get_url(version, arch) == expected


@mark_windows_only
@pytest.mark.parametrize('arch', ['x86_64', 'aarch64'])
def test_get_url_defaults_to_current_arch(set_arch: SetArch, arch: str) -> None:
    set_arch(arch)
    assert install_cxx_toolchain.get_url(
        '4.5'
    ) == install_cxx_toolchain.get_url('4.5', arch)


@mark_windows_only
def test_get_url_unsupported_arch() -> None:
    """RTools 4.4 and later dropped 32-bit builds."""
    assert install_cxx_toolchain.get_url('4.5', 'i686') == ''


@mark_windows_only
@pytest.mark.parametrize(
    'arch,expected', [('x86_64', '4.0'), ('aarch64', '4.5')]
)
def test_latest_version(set_arch: SetArch, arch: str, expected: str) -> None:
    set_arch(arch)
    assert install_cxx_toolchain.latest_version() == expected


@mark_windows_only
@pytest.mark.parametrize(
    'arch,expected', [('x86_64', 'RTools44'), ('aarch64', 'RTools44-aarch64')]
)
def test_toolchain_folder(set_arch: SetArch, arch: str, expected: str) -> None:
    set_arch(arch)
    assert (
        install_cxx_toolchain.get_toolchain_version('RTools', '4.4') == expected
    )


# ---------------------------------------------------------------------------
# is_installed
# ---------------------------------------------------------------------------


@mark_windows_only
@pytest.mark.parametrize('version,arch', sorted(LAYOUTS))
def test_is_installed(
    set_arch: SetArch, tmp_path: Path, version: str, arch: str
) -> None:
    set_arch(arch)
    root = make_toolchain(tmp_path, version, arch)
    assert install_cxx_toolchain.is_installed(root, version)


@mark_windows_only
@pytest.mark.parametrize('version,arch', sorted(LAYOUTS))
def test_is_installed_incomplete(
    set_arch: SetArch, tmp_path: Path, version: str, arch: str
) -> None:
    """Compiler directory exists but the compiler itself is missing."""
    set_arch(arch)
    root = make_toolchain(tmp_path, version, arch, complete=False)
    assert not install_cxx_toolchain.is_installed(root, version)


@mark_windows_only
def test_is_installed_missing_compiler(
    set_arch: SetArch, tmp_path: Path
) -> None:
    """Both directories present, but no compiler binary in them."""
    set_arch('aarch64')
    make_toolchain(tmp_path, '4.4', 'aarch64', complete=False)
    tmp_path.joinpath('usr', 'bin').mkdir(parents=True)
    assert not install_cxx_toolchain.is_installed(str(tmp_path), '4.4')


@mark_windows_only
def test_is_installed_wrong_arch(set_arch: SetArch, tmp_path: Path) -> None:
    """An ARM64 install must not be reported to an x86_64 process."""
    set_arch('aarch64')
    root = make_toolchain(tmp_path, '4.4', 'aarch64')
    set_arch('x86_64')
    assert not install_cxx_toolchain.is_installed(root, '4.4')


@mark_windows_only
def test_is_installed_unknown_version(
    set_arch: SetArch, tmp_path: Path
) -> None:
    set_arch('x86_64')
    root = make_toolchain(tmp_path, '4.4', 'x86_64')
    assert not install_cxx_toolchain.is_installed(root, '9.9')


# ---------------------------------------------------------------------------
# cxx_toolchain_path
# ---------------------------------------------------------------------------


@mark_windows_only
@pytest.mark.parametrize('version,arch', sorted(LAYOUTS))
def test_toolchain_path_from_install_dir(
    set_arch: SetArch, tmp_path: Path, version: str, arch: str
) -> None:
    set_arch(arch)
    folder = install_cxx_toolchain.get_toolchain_version('RTools', version)
    make_toolchain(tmp_path / folder, version, arch)

    compiler_path, tool_path = cxx_toolchain_path(version, str(tmp_path))

    compiler_subdir, tool_subdir, _ = LAYOUTS[(version, arch)]
    assert compiler_path == str(tmp_path.joinpath(folder, *compiler_subdir))
    assert tool_path == str(tmp_path.joinpath(folder, *tool_subdir))
    assert os.environ['PATH'].startswith(f'{compiler_path};{tool_path};')


@mark_windows_only
@pytest.mark.parametrize('version,arch', sorted(LAYOUTS))
def test_toolchain_path_from_env(
    monkeypatch: pytest.MonkeyPatch,
    set_arch: SetArch,
    tmp_path: Path,
    version: str,
    arch: str,
) -> None:
    set_arch(arch)
    root = make_toolchain(tmp_path, version, arch)
    monkeypatch.setenv('CMDSTAN_TOOLCHAIN', root)

    compiler_subdir, tool_subdir, _ = LAYOUTS[(version, arch)]
    assert cxx_toolchain_path() == (
        str(tmp_path.joinpath(*compiler_subdir)),
        str(tmp_path.joinpath(*tool_subdir)),
    )


@mark_windows_only
@pytest.mark.parametrize('arch', ['x86_64', 'aarch64'])
def test_toolchain_path_unversioned_prefers_newest(
    set_arch: SetArch, tmp_path: Path, arch: str
) -> None:
    """With no version requested, 4.5 wins over 4.4."""
    set_arch(arch)
    old = install_cxx_toolchain.get_toolchain_version('RTools', '4.4')
    new = install_cxx_toolchain.get_toolchain_version('RTools', '4.5')
    make_toolchain(tmp_path / old, '4.4', arch)
    make_toolchain(tmp_path / new, '4.5', arch)

    compiler_path, _ = cxx_toolchain_path(None, str(tmp_path))
    assert new in compiler_path


@mark_windows_only
def test_toolchain_path_rtools_home(
    monkeypatch: pytest.MonkeyPatch, set_arch: SetArch, tmp_path: Path
) -> None:
    set_arch('x86_64')
    root = make_toolchain(tmp_path, '4.4', 'x86_64')
    monkeypatch.setenv('RTOOLS44_HOME', root)
    compiler_path, _ = cxx_toolchain_path()
    assert compiler_path.startswith(root)


@mark_windows_only
def test_toolchain_path_incomplete_warns(
    monkeypatch: pytest.MonkeyPatch,
    set_arch: SetArch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    set_arch('x86_64')
    root = make_toolchain(tmp_path, '4.4', 'x86_64', complete=False)
    monkeypatch.setenv('CMDSTAN_TOOLCHAIN', root)
    with pytest.raises(ValueError, match='no RTools toolchain installation'):
        cxx_toolchain_path()
    assert 'Found invalid RTools installation' in caplog.text


@mark_windows_only
def test_toolchain_path_unrecognized_root(
    monkeypatch: pytest.MonkeyPatch,
    set_arch: SetArch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A directory that exists but matches no known layout."""
    set_arch('x86_64')
    monkeypatch.setenv('CMDSTAN_TOOLCHAIN', str(tmp_path))
    with pytest.raises(ValueError, match='no RTools toolchain installation'):
        cxx_toolchain_path()
    assert 'Found no usable RTools installation' in caplog.text


@mark_windows_only
def test_toolchain_path_not_found(
    monkeypatch: pytest.MonkeyPatch, set_arch: SetArch, tmp_path: Path
) -> None:
    set_arch('x86_64')
    monkeypatch.setenv('CMDSTAN_TOOLCHAIN', str(tmp_path / 'nowhere'))
    with pytest.raises(ValueError, match='no RTools toolchain installation'):
        cxx_toolchain_path()


@mark_windows_only
def test_toolchain_path_unsupported_version(set_arch: SetArch) -> None:
    set_arch('aarch64')
    with pytest.raises(ValueError, match='unsupported RTools version'):
        cxx_toolchain_path('4.0')


@mark_windows_only
def test_toolchain_path_bad_version_type() -> None:
    with pytest.raises(TypeError, match='Format version number as a string'):
        cxx_toolchain_path(4.0)  # type: ignore[arg-type]


@mark_not_windows
def test_cxx_toolchain_path_not_windows() -> None:
    with pytest.raises(RuntimeError, match='only supported on Windows'):
        cxx_toolchain_path()


# ---------------------------------------------------------------------------
# misc
# ---------------------------------------------------------------------------


@mark_windows_only
def test_config() -> None:
    """Test config output."""

    config = install_cxx_toolchain.get_config('C:\\RTools', True)

    config_reference = [
        '/SP-',
        '/VERYSILENT',
        '/SUPPRESSMSGBOXES',
        '/CURRENTUSER',
        'LANG="English"',
        '/DIR="RTools"',
        '/NOICONS',
        '/NORESTART',
    ]

    assert config == config_reference


@mark_not_windows
def test_install_not_windows() -> None:
    """Try to install on unsupported platform."""

    with pytest.raises(
        NotImplementedError,
        match=r'Download for the C\+\+ toolchain on the current platform '
        r'has not been implemented:\s*\S+',
    ):
        install_cxx_toolchain.run_rtools_install({})


@mark_windows_only
def test_install_unsupported_combination(
    set_arch: SetArch, tmp_path: Path
) -> None:
    """RTools 4.0 has no ARM64 build."""
    set_arch('aarch64')
    with pytest.raises(ValueError, match='not available for aarch64'):
        install_cxx_toolchain.run_rtools_install(
            {'version': '4.0', 'dir': str(tmp_path)}
        )


@mark_windows_only
def test_install_defaults_to_latest_version(
    monkeypatch: pytest.MonkeyPatch, set_arch: SetArch, tmp_path: Path
) -> None:
    set_arch('x86_64')
    monkeypatch.setattr(install_cxx_toolchain, 'latest_version', lambda: '9.9')
    with pytest.raises(ValueError, match='RTools 9.9 is not available'):
        install_cxx_toolchain.run_rtools_install(
            {'version': None, 'dir': str(tmp_path)}
        )


@mark_windows_only
def test_usage(capsys: pytest.CaptureFixture) -> None:
    install_cxx_toolchain.usage()
    assert '--version' in capsys.readouterr().out


def test_parse_cmdline_args(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sys, 'argv', ['install_cxx_toolchain', '-v', '4.5', '-d', 'somewhere']
    )
    args = install_cxx_toolchain.parse_cmdline_args()
    assert args['version'] == '4.5'
    assert args['dir'] == 'somewhere'
    assert args['silent'] is False
    assert args['progress'] is False
