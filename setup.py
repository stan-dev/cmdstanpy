#!/usr/bin/env python

import setuptools


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
Programming Language :: Other
Programming Language :: C++
Topic :: Scientific/Engineering :: Information Analysis
"""

setuptools.setup(
    name='pycmdstan',
    version='0.9',
    description='Python interface to CmdStan',
    long_description=readme_contents(),
    long_description_content_type="text/markdown",
    author='Marmaduke Woodman',
    author_email='marmaduke.woodman@univ-amu.fr',
    url='https://gitlab.thevirtualbrain.org/tvb/pycmdstan',
    packages=['pycmdstan'],
    install_requires='numpy filelock matplotlib'.split(),
    classifiers=_classifiers.strip().split('\n'),
)
