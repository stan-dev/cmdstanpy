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
and model and data.  It manages the set of CmdStan input and output files.
It runs and records an analysis, but the user chooses
whether or not to instantiate the results in memory,
thus CmdStanPy has the potential to fit more complex models
to larger datasets than might be possible in PyStan or RStan.

The statistical modeling enterprise has two principal modalities:
development and production.
The focus of development is model building, comparison, and validation.
Many models are written and fitted to many kinds of data.
The focus of production is using a known model on real-world data in order
obtain estimates used for decision-making.
CmdStanPy is designed to support both of these modalities.
Because model development and testing may require many iterations,
the defaults favor development mode and therefore output files are
stored on a temporary filesystem.
Non-default options allow all aspects of a run to be specified
so that scripts can be used to distributed analysis jobs across
nodes and machines.

