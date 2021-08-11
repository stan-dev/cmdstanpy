#############
API Reference
#############

*******
Classes
*******

.. _class_cmdstanmodel:

CmdStanModel
============

A CmdStanModel object encapsulates the Stan program. It manages program compilation and provides the following inference methods:

sample
    runs the HMC-NUTS sampler to produce a set of draws from the posterior distribution.

optimize
    produce a penalized maximum likelihood estimate (point estimate) of the model parameters.

variational
    run CmdStan’s variational inference algorithm to approximate the posterior distribution.

generate_quantities
    runs CmdStan’s generate_quantities method to produce additional quantities of interest based on draws from an existing sample.

.. autoclass:: cmdstanpy.CmdStanModel
   :members:


.. _class_cmdstanmcmc:

CmdStanMCMC
===========

.. autoclass:: cmdstanpy.CmdStanMCMC
   :members:

.. _class_cmdstanmle:

CmdStanMLE
==========

.. autoclass:: cmdstanpy.CmdStanMLE
   :members:

.. _class_cmdstanqq:

CmdStanGQ
=========

.. autoclass:: cmdstanpy.CmdStanGQ
   :members:

.. _class_cmdstanvb:

CmdStanVB
=========

.. autoclass:: cmdstanpy.CmdStanVB
   :members:

.. _class_runset:

RunSet
======

.. autoclass:: cmdstanpy.stanfit.RunSet
   :members:

*********
Functions
*********

.. _function_cmdstan_path:

cmdstan_path
============

.. autofunction:: cmdstanpy.cmdstan_path

.. _function_install_cmdstan:

install_cmdstan
===============

.. autofunction:: cmdstanpy.install_cmdstan

.. _function_set_cmdstan_path:

set_cmdstan_path
================

.. autofunction:: cmdstanpy.set_cmdstan_path

.. _function_set_make_env:

set_make_env
============

.. autofunction:: cmdstanpy.set_make_env
