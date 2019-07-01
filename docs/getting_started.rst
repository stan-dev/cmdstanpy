Getting Started
===============


Installation
____________

CmdStanPy is a pure-Python package which can be installed from PyPI

.. code-block:: bash

    pip install --upgrade cmdstanpy

or from GitHub

.. code-block:: bash

    pip install -e git+https://github.com/stan-dev/cmdstanpy#egg=cmdstanpy

CmdStanPy requires a local install of CmdStan.
If you don't have CmdStan installed, you can run the CmdStanPy script ``install_cmdstan``
which downloads CmdStan from GitHub and builds the CmdStan utilities.
By default this script installs the latest version of CmdStan into a directory named
``.cmdstanpy`` in your ``$HOME`` directory:

.. code-block:: bash

    install_cmdstan
    ls -F ~/.cmdstanpy

The named arguments: `-d <directory>` and  `-v <version>`
can be used to override these defaults:

.. code-block:: bash

    install_cmdstan -d my_local_cmdstan -v 2.19.1
    ls -F my_local_cmdstan

If you already have CmdStan installed in a directory
then you can set the environment variable ``CMDSTAN`` to this
location and it will be picked up by CmdStanPy:

.. code-block:: bash

    export CMDSTAN='/path/to/cmdstan-2.19.1'


The CmdStanPy commands ``cmdstan_path`` and ``set_cmdstan_path``
get and set this environment variable:

.. code-block:: python

    from cmdstanpy import cmdstan_path, set_cmdstan_path

    oldpath = cmdstan_path()
    set_cmdstan_path(os.join('path','to','cmdstan'))
    newpath = cmdstan_path()



CmdStanPy's "Hello World"
_________________________

To exercise the essential functions of CmdStanPy, we will
compile and run the example Stan model ``bernoulli.stan`` which is
distributed with CmdStan.


Specify a Stan model
--------------------

The ``Model`` class specifies the Stan program and its corresponding compiled executable.
The function ``compile_model`` is used to compile or or recompile a Stan program.
It takes the path to the Stan program file and returns a ``Model`` object:

.. code-block:: python

    import os
    import os.path
    from cmdstanpy import Model, compile_model

    bernoulli_stan = os.path.join(cmdstan_path(), 'examples', 'bernoulli', 'bernoulli.stan')
    bernoulli_model = compile_model(bernoulli_stan)

If you already have a compiled executable, you can construct a ``Model`` object directly:

.. code-block:: python

    bernoulli_model = Model(
            stan_file=os.path.join(cmdstan_path(), 'examples', 'bernoulli', 'bernoulli.stan')
            stan_exe=os.path.join(cmdstan_path(), 'examples', 'bernoulli', 'bernoulli')
            )


Run the HMC-NUTS sampler
------------------------

The ``sample`` function invokes the Stan HMC-NUTS sampler on the ``Model`` object and some data
and returns a ``StanFit`` object:

.. code-block:: python

    from cmdstanpy import sample, StanFit

    bern_data = { "N" : 10, "y" : [0,1,0,0,0,0,0,0,0,1] }
    bern_fit = sample(bernoulli_model, data=bern_data)
    
By default, the ``sample`` command runs 4 sampler chains.
The ``StanFit`` object records the results of each sampler chain.
If no output file path is specified, the sampler outputs
are written to a temporary directory which is deleted
when the current Python session is terminated.


Summarize or save the results
-----------------------------

The ``get_drawset`` function is used to get the draws from
all chains as a ``pandas.DataFrame``, one draw per row, one column per
model parameter, transformed parameter, generated quantity variable.
The ``params`` argument is used to restrict the DataFrame
columns to just the specified parameter names.

.. code-block:: python

    get_drawset(bern_fit, params=['theta'])

Underlyingly, this information is stored in the ``sample`` property
of a ``StanFit`` object as a 3-D ``numpy.ndarray`` (i.e., a multi-dimensional array)
with dimensions: (draws, chains, columns).
Python's index slicing operations can be used to access the information by chain.
For example, to select all draws and all output columns from the first chain,
we specify the chain index (2nd index dimension).  As arrays indexing starts at 0,
the index '0' corresponds to the first chain in the ``StanFit``:

.. code-block:: python

    chain_1 = bern_fit.sample[:,0,:]


CmdStan is distributed with a posterior analysis utility ``stansummary``
that reads the outputs of all chains and computes summary statistics
on the model fit for all parameters. CmdStanPy's ``summary`` function
runs the CmdStan ``stansummary`` utility and returns the output as a pandas.DataFrame:

.. code-block:: python

    from cmdstanpy import summary

    summary(bern_fit)

CmdStan is distributed with a second posterior analysis utility ``diagnose``
that reads the outputs of all chains and checks for the following
potential problems:

+ Transitions that hit the maximum treedepth
+ Divergent transitions
+ Low E-BFMI values (sampler transitions HMC potential energy)
+ Low effective sample sizes
+ High R-hat values

The ``diagnose`` function runs the CmdStan ``diagnose`` utility
and prints the output to the console.

.. code-block:: python

    from cmdstanpy import diagnose

    diagnose(bern_fit)

By default, CmdStanPy will save all CmdStan outputs in a temporary
directory which is deleted when the Python session exits.
In particular, unless the ``csv_output_file`` argument to the ``sample``
function is overtly specified, all the csv output files will be written into
this temporary directory and then when the session exits.
The ``save_csvfiles`` function moves the CmdStan csv output files
to the specified location, renaming them using a specified basename.

.. code-block:: python

    from cmdstanpy import save_csvfiles

    save_csvfiles(bern_fit, dir='some/path', basename='descriptive-name')
