.. py:currentmodule:: cmdstanpy

#############
API Reference
#############

The following documents the public API of CmdStanPy. It is expected to be stable between versions,
with backwards compatibility between minor versions and deprecation warnings preceding breaking changes.
The documentation for the `internal API <internal_api.rst>`_ is also provided, but the internal API
does not guarantee either stability and backwards compatibility.

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
    produce a penalized maximum likelihood estimate or maximum a posteriori estimate (point estimate) of the model parameters.

:meth:`~CmdStanModel.laplace_sample`
    draw from a Laplace approximatation centered at the posterior mode found by ``optimize``.

:meth:`~CmdStanModel.pathfinder`
    runs the Pathfinder variational inference parameters to recieve approximate draws from the posterior.

:meth:`~CmdStanModel.variational`
    run CmdStan’s automatic differentiation variational inference (ADVI) algorithm to approximate the posterior distribution.

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

CmdStanLaplace
==============

.. autoclass:: cmdstanpy.CmdStanLaplace
   :members:

CmdStanPathfinder
=================

.. autoclass:: cmdstanpy.CmdStanPathfinder
   :members:

CmdStanVB
=========

.. autoclass:: cmdstanpy.CmdStanVB
   :members:

CmdStanGQ
=========

.. autoclass:: cmdstanpy.CmdStanGQ
   :members:

*********
Functions
*********

compile_stan_file
=================

.. autofunction:: cmdstanpy.compile_stan_file


format_stan_file
================

.. autofunction:: cmdstanpy.format_stan_file

show_versions
=============

.. autofunction:: cmdstanpy.show_versions

cmdstan_path
============

.. autofunction:: cmdstanpy.cmdstan_path

install_cmdstan
===============

.. autofunction:: cmdstanpy.install_cmdstan

rebuild_cmdstan
===============

.. autofunction:: cmdstanpy.rebuild_cmdstan

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
