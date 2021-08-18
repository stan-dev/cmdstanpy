Overview
========

CmdStanPy is a lightweight interface to Stan for Python users which
provides the necessary objects and functions to do Bayesian inference
given a probability model and data.
It wraps the
`CmdStan <https://mc-stan.org/docs/cmdstan-guide/cmdstan-installation.html>`_
command line interface in a small set of
Python classes which provide methods to do analysis and manage the resulting
set of model, data, and posterior estimates.

CmdStanPy is a lightweight interface in that it is designed to use minimal
memory beyond what is used by CmdStan itself to do inference given
and model and data.It runs and records an analysis, but the user chooses
whether or not to instantiate the results in memory,
thus CmdStanPy has the potential to fit more complex models
to larger datasets than might be possible in PyStan or RStan.
It manages the set of CmdStan input and output files and provides
methods and options which allow the user to save these files
to a specific filepath.
By default, CmdStan output files are written to a temporary directory
in order to avoid filling up the user's filesystem.

