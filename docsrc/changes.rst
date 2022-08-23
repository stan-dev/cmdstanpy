.. py:currentmodule:: cmdstanpy

.. NOTE FOR MAINTAINERS: This should be updated just before the release action is run, not every PR.

What's New
==========

For full changes, see the `Releases page <https://github.com/stan-dev/cmdstanpy/releases>`__ on GitHub.


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
