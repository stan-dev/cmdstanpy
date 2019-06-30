#!/usr/bin/env python

import setuptools
from cmdstanpy import __version__


def readme_contents() -> str:
    with open('README.md', 'r') as fd:
        src = fd.read()
    return src


_classifiers = """
Programming Language :: Python :: 3
License :: OSI Approved :: Apache Software License
Operating System :: OS Independent
Development Status :: 4 - Beta
Intended Audience :: Science/Research
Natural Language :: English
Programming Language :: Python
Topic :: Scientific/Engineering :: Information Analysis
"""

INSTALL_REQUIRES = ['numpy', 'pandas']

EXTRAS_REQUIRE = {
    'tests': ['pytest', 'pytest-cov'],
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
    version=__version__,
    description='Python interface to CmdStan',
    long_description=readme_contents(),
    long_description_content_type="text/markdown",
    author='Stan Dev Team',
    url='https://github.com/stan-dev/cmdstanpy',
    packages=['cmdstanpy'],
    scripts=['bin/install_cmdstan'],
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
    classifiers=_classifiers.strip().split('\n'),
)
