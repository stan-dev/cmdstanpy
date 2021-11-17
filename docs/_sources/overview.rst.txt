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
It is lightweight in that it uses minimal
memory beyond the memory used by CmdStan.
CmdStanPy runs CmdStan, but only instantiates the resulting inference
objects in memory upon request.
Thus CmdStanPy has the potential to fit more complex models
to larger datasets than might be possible in PyStan or RStan.

CmdStan is a file-based interface.
CmdStanPy manages the Stan program files and the CmdStan output files.
The user can specify the output directory for the CmdStan outputs,
otherwise the files will be written to a 
temporary filesystem which persists throughout the session.
This allows the user to test and develop models prospectively,
following the Bayesian workflow.



