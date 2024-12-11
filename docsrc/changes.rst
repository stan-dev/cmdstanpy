.. py:currentmodule:: cmdstanpy

.. NOTE FOR MAINTAINERS: This should be updated just before the release action is run, not every PR.

What's New
==========

For full changes, see the `Releases page <https://github.com/stan-dev/cmdstanpy/releases>`_ on GitHub.

CmdStanPy 1.2.5
---------------

- Fixed issues that arose when running a model with no parameters using CmdStan 2.36.0+

Reminder: The next non-bugfix release of CmdStanPy will be version 2.0, which will remove all existing deprecations.

CmdStanPy 1.2.4
---------------

- Fixed a bug in `from_csv` which prevented reading files created by CmdStan 2.35.0+

Reminder: The next non-bugfix release of CmdStanPy will be version 2.0, which will remove all existing deprecations.

CmdStanPy 1.2.3
---------------

- Updated the logic around reading Stan CSV files to support CmdStan 2.35.0+
- Fixed an issue where the ``profile_files`` member of the RunSet object was not correct when running multiple chains in the same process.

Reminder: The next non-bugfix release of CmdStanPy will be version 2.0, which will remove all existing deprecations.

CmdStanPy 1.2.2
---------------

- Updated Community page to link to the ``bibat`` package.
- Moved CmdStanPy's metadata to exclusively use the ``pyproject.toml`` file.
- Fixed an issue where the deprecation of the ``compile=False`` argument to :class:`CmdStanModel` would
  make it impossible to use the canonicalizer to update old models.
  The new function :func:`cmdstanpy.format_stan_file` supports this use case.
- Fixed a bug preventing multiple inits from being used with :meth:`CmdStanModel.pathfinder`.
- Added a helper argument ``num_threads`` to :meth:`CmdStanModel.pathfinder`.

Reminder: The next non-bugfix release of CmdStanPy will be version 2.0, which will remove all existing deprecations.

CmdStanPy 1.2.1
---------------

- Switched from :class:`numpy.random.RandomState` to :func:`~numpy.random.default_rng`.
- Fixed minor doc typos.
- Stan 2.34: Fixed parsing of unit_e output files.
- Stan 2.34: Exposed new Pathfinder arguments.
- Allow the ``vars`` argument to :meth:`CmdStanMCMC.draws_pd` to filter the ``chain__``, ``iter__``, and ``draws__`` columns.
- Fixed a bug involving the interaction of the ``adapt_engaged`` and ``iter_warmup`` arguments to :meth:`CmdStanMCMC.sample`.

Reminder: The next non-bugfix release of CmdStanPy will be version 2.0, which will remove all existing deprecations.

CmdStanPy 1.2.0
---------------
- **New functionality**

  - The Pathfinder algorithm (available in CmdStan 2.33+) is now availble as :meth:`CmdStanModel.pathfinder`.
  - Laplace approximations (available in CmdStan 2.32+) are now available as :meth:`CmdStanModel.laplace_sample`.
  - The :meth:`CmdStanModel.optimize` method now supports the ``jacobian`` boolean argument to enable change-of-variables adjustments.
    When enabled, the Maximum a posteriori estimate (MAP) is returned, rather than the MLE.
  - The :func:`cmdstanpy.install_cmdstan` function and script can install development versions of CmdStan using the ``git:`` prefix in the version.

- **Deprecations**
  The next non-bugfix release of CmdStanPy will be version 2.0, which will remove all existing deprecations. Additional deprecations in this version:

  - :class:`CmdStanModel` will *require* that it has a compiled executable after construction. The ``compile`` argument is deprecated,
    (the ability to force recompilation is available under the argument ``force_compile``), and the ``compile()`` method is deprecated.
    If you wish to compile Stan files independent of constructing a model, use :func:`cmdstanpy.compile_stan_file`.
  - :meth:`CmdStanMLE.stan_variable` will begin returning a :class:`np.ndarray` in all cases, as opposed to the current behavior where sometimes a float is returned.
  - :meth:`CmdStanVB.stan_variables` will return the _draws_ from the approximate posterior, rather than the optimized mean.
    A new argument, ``mean``, can be set to True to return the mean instead. Additionally, a :class:`np.ndarray` will be returned in all cases starting in the next version.
  - :meth:`CmdStanModel.variational` argument ``output_samples`` will has been renamed to ``draws``.

- **Other changes**

  - A list of dictionaries is now allowed as the ``inits`` argument to :meth:`CmdStanModel.sample`.
  - :func:`cmdstanpy.install_cmdstan` correctly fetches the CmdStan version for ppc64el machines.
  - The documentation on how to use external C++ code was updated.
  - Various other bug fixes.

.. note::
    The minimum supported version for CmdStanPy is now Python 3.8.

CmdStanPy 1.1.0
---------------
- **New functionality**

  - :meth:`CmdStanModel.generate_quantities` can now accept samples from optimization and variational inference.
    The argument ``mcmc_sample`` has been renamed ``previous_fit`` to reflect this; the former name is still accepted
    but deprecated.
  - :meth:`CmdStanModel.log_prob` is able to return the log probability and its gradient with respect to a set of parameters.
    **Note** that this is *not* an efficient way of calculating this in general and should be reserved for debugging
    and model development.
- **Other changes**

  - Improved some of the type hints in the package.
  - Ensure draws are serialized if a fit object is pickled.
  - :meth:`~CmdStanModel.src_info` now raises an error if the command fails, rather than returning ``{}``.
  - CmdStanPy has transitioned all unit tests from the ``unittest`` library to use ``pytest``.

CmdStanPy 1.0.8
---------------

- ``install_cmdstan`` now downloads the correct CmdStan for non-x86 Linux machines.
- Improved reporting of errors during :meth:`~CmdStanModel.compile`.
- Fixed some edge cases in mixing arguments of the :meth:`~CmdStanModel.optimize` function.
- Fixed how ``NaN`` and infinite numbers were serialized to JSON.
- Removed dependency on ``ujson``. For now, all JSON serialization is done with the Python standard library.
- Added a ``timeout`` parameter to all model methods which can be used to terminate the CmdStan process after the specified time.
- A model will now properly recompile if one of the `#include`-d files changed since it was last built.

CmdStanPy 1.0.7
---------------

- Fixed an issue where complex number containers in Stan program outputs were not being read in properly by CmdStanPy. The output would have the correct shape, but the values would be mixed up.

CmdStanPy 1.0.6
---------------

- Fixed a build error in the documentation
- Improved messages when model fails to compile due to C++ errors.

CmdStanPy 1.0.5
---------------

- Fixed a typo in :func:`cmdstanpy.show_versions()`
- Reorganized and updated the documentation
- Reorganized a lot of internal code
- Cleaned up the output of :meth:`CmdStanMCMC.draws_pd`
- Cleaned up the output of :meth:`CmdStanMCMC.summary`
- Removed the logging which occurred when Python exited with cmdstanpy imported.

CmdStanPy 1.0.4
---------------

- Fix an issue with :func:`cmdstanpy.install_cmdstan()` where the installation would report that it had failed even when it had not.

CmdStanPy 1.0.3
---------------

- Fix an issue where Stan fit objects were not ``pickle``-able when they previously were.

  .. warning::
      We still do not recommend pickling cmdstanpy objects, but rather using functions :meth:`~CmdStanMCMC.save_csvfiles` and :func:`~cmdstanpy.from_csv`.

CmdStanPy 1.0.2
---------------

- CmdStanPy can now format (and canonicalize) your Stan files with :meth:`CmdStanModel.format()`
- Stan variables can now be accessed from fit objects using the `.` syntax when no naming conflicts occur. For example, previous code ``fit.stan_variable("my_cool_variable")`` can now be written ``fit.my_cool_variable``
- CmdStanPy is more robust to running in threaded environments and tries harder to not overwrite its own output files
- The ``install_cmdstan`` script can now be run in interactive mode using ``--interactive``/``-i``
- CmdStanPy now computes some diagnostics after running HMC and will warn you about post-warmup divergences and treedepth exceptions
- Runtime exceptions in the ``generated quantities`` block should be recognized better now.
- The default level of precision used by :meth:`CmdStanMCMC.summary()` is now 6, as it is when ``stansummary`` is used from the command line.\
- Various documentation improvements


CmdStanPy 1.0.1
---------------

- Support new optimizations in CmdStan 2.29
- Support complex numbers as both inputs and outputs of Stan programs
- Sped up assembling output by only reading draws at most once
- Fixed an issue where a command failing could change your working directory
- Improve error messages in some cases
- CmdStanPy no longer changes the global root logging level

.. note::
    The minimum supported version for CmdStanPy is now Python 3.7.


CmdStanPy 1.0.0
---------------

- Initial release
