.. CmdStanPy documentation master file, created by
   sphinx-quickstart on Wed Jun  6 13:32:52 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

===============================================
:mod:`cmdstanpy` -- Python interface to CmdStan
===============================================

.. module:: cmdstanpy
   :synopsis: A lightweight pure-Python interface to CmdStan which provides access to the Stan compiler and all inference algorithms.

.. moduleauthor:: Stan Developement Team

CmdStanPy is a lightweight interface to Stan for Python users which
provides the necessary objects and functions to do Bayesian inference
given a probability model and data.
It wraps the
`CmdStan <https://mc-stan.org/docs/cmdstan-guide/cmdstan-installation.html>`_
command line interface in a small set of
Python classes which provide methods to do analysis and manage the resulting
set of model, data, and posterior estimates.

.. toctree::
   :maxdepth: 4

   overview
   installation
   hello_world
   workflow
   examples
   community
   api

:ref:`genindex`
