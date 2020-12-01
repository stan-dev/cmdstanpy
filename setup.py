#!/usr/bin/env python

import os
import re

import setuptools

HERE = os.path.dirname(__file__)


def readme_contents() -> str:
    with open(os.path.join(HERE, 'README.md'), 'r') as fd:
        src = fd.read()
    return src


def requirements() -> str:
    with open(os.path.join(HERE, 'requirements.txt'), 'r') as fd:
        src = fd.read()
    return src.splitlines()


def requirements_test() -> str:
    with open(os.path.join(HERE, 'requirements-test.txt'), 'r') as fd:
        src = fd.read()
    return src.splitlines()


def requirements_optional() -> str:
    with open(os.path.join(HERE, 'requirements-optional.txt'), 'r') as fd:
        src = fd.read()
    return src.splitlines()


def get_version() -> str:
    version_file = open(os.path.join('cmdstanpy', '_version.py'))
    version_contents = version_file.read()
    return re.search("__version__ = '(.*?)'", version_contents).group(1)


_classifiers = """
Programming Language :: Python :: 3
License :: OSI Approved :: BSD License
Operating System :: OS Independent
Development Status :: 4 - Beta
Intended Audience :: Science/Research
Natural Language :: English
Programming Language :: Python
Topic :: Scientific/Engineering :: Information Analysis
"""

INSTALL_REQUIRES = requirements()

EXTRAS_REQUIRE = {
    'all': requirements_optional(),
    'tests': requirements_test(),
    'docs': [
        'sphinx',
        'sphinx-gallery',
        'sphinx_rtd_theme',
        'numpydoc',
        'matplotlib',
    ],
}

setuptools.setup(
    name='cmdstanpy',
    version=get_version(),
    description='Python interface to CmdStan',
    long_description=readme_contents(),
    long_description_content_type="text/markdown",
    author='Stan Dev Team',
    url='https://github.com/stan-dev/cmdstanpy',
    packages=['cmdstanpy'],
    entry_points={
        'console_scripts': [
            'install_cmdstan=cmdstanpy.install_cmdstan:main',
            'install_cxx_toolchain=cmdstanpy.install_cxx_toolchain:main',
        ]
    },
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
    classifiers=_classifiers.strip().split('\n'),
)
