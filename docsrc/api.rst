.. py:currentmodule:: cmdstanpy

#############
API Reference
#############

The following documents the public API of CmdStanPy. It is expected to be stable between versions,
with backwards compatibility between minor versions and deprecation warnings preceeding breaking changes.
There is also the `internal API <internal_api.rst>`__, which is makes no such guarantees.

.. toctree::
   :hidden:

   internal_api.rst

*******
Classes
*******

CmdStanModel
============

A CmdStanModel object encapsulates the Stan program. It manages program compilation and provides the following inference methods:

:meth:`~CmdStanModel.sample`
    runs the HMC-NUTS sampler to produce a set of draws from the posterior distribution.

:meth:`~CmdStanModel.optimize`
    produce a penalized maximum likelihood estimate (point estimate) of the model parameters.

:meth:`~CmdStanModel.variational`
    run CmdStan’s variational inference algorithm to approximate the posterior distribution.

:meth:`~CmdStanModel.generate_quantities`
    runs CmdStan’s generate_quantities method to produce additional quantities of interest based on draws from an existing sample.

.. autoclass:: cmdstanpy.CmdStanModel
   :members:


CmdStanMCMC
===========

.. autoclass:: cmdstanpy.CmdStanMCMC
   :members:

CmdStanMLE
==========

.. autoclass:: cmdstanpy.CmdStanMLE
   :members:

CmdStanGQ
=========

.. autoclass:: cmdstanpy.CmdStanGQ
   :members:

CmdStanVB
=========

.. autoclass:: cmdstanpy.CmdStanVB
   :members:


*********
Functions
*********

show_versions
=============

.. autofunction:: cmdstanpy.show_versions

cmdstan_path
============

.. autofunction:: cmdstanpy.cmdstan_path

install_cmdstan
===============

.. autofunction:: cmdstanpy.install_cmdstan

set_cmdstan_path
================

.. autofunction:: cmdstanpy.set_cmdstan_path

cmdstan_version
================

.. autofunction:: cmdstanpy.cmdstan_version

set_make_env
============

.. autofunction:: cmdstanpy.set_make_env

from_csv
========

.. autofunction:: cmdstanpy.from_csv

write_stan_json
===============

.. autofunction:: cmdstanpy.write_stan_json
