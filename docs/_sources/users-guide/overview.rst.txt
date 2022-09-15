Overview
========

CmdStanPy is a lightweight interface to `Stan <https://mc-stan.org/docs/stan-users-guide/index.html>`_ for Python.
It provides a small set of classes and methods for doing Bayesian inference
given a probability model and data.
With CmdStanPy, you can:

+ Compile a Stan model.

+ Do inference on the model conditioned on the data, using one of Stan inference algorithms

    + Exact Bayesian estimation using the `NUTS-HMC sampler <https://mc-stan.org/docs/reference-manual/hmc.html>`_.

    + Approximate Bayesian estimation using `ADVI <https://mc-stan.org/docs/reference-manual/vi-algorithms.html>`_.

    + MAP estimation by `optimization <https://mc-stan.org/docs/reference-manual/optimization-algorithms.html>`_.

+ Generate new quantities of interest from a model given an existing sample.

+ Manage the resulting inference engine outputs:  extract information, summarize results, and save the outputs.

CmdStanPy wraps the
`CmdStan <https://mc-stan.org/docs/cmdstan-guide/cmdstan-installation.html>`_
file-based command line interface.
It is lightweight in that it uses minimal memory beyond the memory used by CmdStan,
thus CmdStanPy has the potential to fit more complex models
to larger datasets than might be possible in PyStan or RStan.

CmdStanPy is designed to support the development, testing, and deployment of a Stan model.
CmdStanPy manages the Stan program files, data files, and CmdStan output files.
By default, output files are written to a temporary filesystem which persists
throughout the session.  This is appropriate behavior during model development
because it allows the user to test many models without filsystem clutter or worse.
Once deployed into production, 
the user can specify the output directory for the CmdStan outputs.


