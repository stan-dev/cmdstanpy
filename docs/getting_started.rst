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

To install CmdStanPy with all the optional packages
(ujson; json processing, tqdm; progress bar)

.. code-block:: bash

    pip install --upgrade cmdstanpy[all]

CmdStanPy requires a local install of CmdStan.
If you don't have CmdStan installed, you can run the CmdStanPy script ``install_cmdstan``
which downloads CmdStan from GitHub and builds the CmdStan utilities.
By default this script installs the latest version of CmdStan into a directory named
``.cmdstanpy`` in your ``$HOME`` directory:

.. code-block:: bash

    install_cmdstan
    ls -F ~/.cmdstanpy

On Windows

.. code-block:: bash

    python -m cmdstanpy.install_cmdstan
    dir "%HOME%/.cmdstanpy"

The named arguments: `-d <directory>` and  `-v <version>`
can be used to override these defaults:

.. code-block:: bash

    install_cmdstan -d my_local_cmdstan -v 2.20.0
    ls -F my_local_cmdstan

If you already have CmdStan installed in a directory
then you can set the environment variable ``CMDSTAN`` to this
location and it will be picked up by CmdStanPy:

.. code-block:: bash

    export CMDSTAN='/path/to/cmdstan-2.20.0'


The CmdStanPy commands ``cmdstan_path`` and ``set_cmdstan_path``
get and set this environment variable:

.. code-block:: python

    from cmdstanpy import cmdstan_path, set_cmdstan_path

    oldpath = cmdstan_path()
    set_cmdstan_path(os.path.join('path','to','cmdstan'))
    newpath = cmdstan_path()

To use custom ``make``-tool use ``set_make_env`` function.

.. code-block:: python

    from cmdstanpy import set_make_env
    set_make_env("mingw32-make.exe") # On Windows with mingw32-make

User can install optional packages with pip with the CmdStanPy installation

.. code-block:: bash

    pip install --upgrade cmdstanpy[all]

or by installing packages manually.

For faster IO cmdstanpy will use ``ujson`` package if it's installed

.. code-block:: bash

    pip install ujson

To enable progress bar user can install ``tqdm`` package

.. code-block:: bash

    pip install tqdm


CmdStanPy's "Hello World"
_________________________

To exercise the essential functions of CmdStanPy, we will
compile and run the example Stan model ``bernoulli.stan`` which is
distributed with CmdStan.


Specify a Stan model
--------------------

The ``Model`` class specifies the Stan program and its corresponding compiled executable.
The method ``compile`` is used to compile or or recompile a Stan program.

.. code-block:: python

    import os
    from cmdstanpy import cmdstan_path, Model, StanFit

    bernoulli_stan = os.path.join(cmdstan_path(), 'examples', 'bernoulli', 'bernoulli.stan')
    bernoulli_model = Model(stan_file=bernoulli_stan)
    bernoulli_model.compile()

If you already have a compiled executable, you can construct a ``Model`` object directly:

.. code-block:: python

    bernoulli_model = Model(
            stan_file=os.path.join(cmdstan_path(), 'examples', 'bernoulli', 'bernoulli.stan')
            stan_exe=os.path.join(cmdstan_path(), 'examples', 'bernoulli', 'bernoulli')
            )


Run the HMC-NUTS sampler
------------------------

The ``Model`` method ``sample`` runs the Stan HMC-NUTS sampler on the model and data
and returns a ``StanFit`` object:

.. code-block:: python

    bernoulli_data = { "N" : 10, "y" : [0,1,0,0,0,0,0,0,0,1] }
    bern_fit = bernoulli_model.sample(data=bernoulli_data)

By default, the ``sample`` command runs 4 sampler chains.
The ``StanFit`` object records the results of each sampler chain.
If no output file path is specified, the sampler outputs
are written to a temporary directory which is deleted
when the current Python session is terminated.


Summarize or save the results
-----------------------------

The ``get_drawset`` method returns the draws from
all chains as a ``pandas.DataFrame``, one draw per row, one column per
model parameter, transformed parameter, generated quantity variable.
The ``params`` argument is used to restrict the DataFrame
columns to just the specified parameter names.

.. code-block:: python

    bern_fit.get_drawset(params=['theta'])

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
on the model fit for all parameters. The ``StanFit`` method ``summary``
runs the CmdStan ``stansummary`` utility and returns the output as a pandas.DataFrame:

.. code-block:: python

    bern_fit.summary()

CmdStan is distributed with a second posterior analysis utility ``diagnose``
that reads the outputs of all chains and checks for the following
potential problems:

+ Transitions that hit the maximum treedepth
+ Divergent transitions
+ Low E-BFMI values (sampler transitions HMC potential energy)
+ Low effective sample sizes
+ High R-hat values

The ``StanFit`` method ``diagnose`` runs the CmdStan ``diagnose`` utility
and prints the output to the console.

.. code-block:: python

    bern_fit.diagnose()

By default, CmdStanPy will save all CmdStan outputs in a temporary
directory which is deleted when the Python session exits.
In particular, unless the ``csv_basename`` argument to the ``sample``
function is overtly specified, all the csv output files will be written into
this temporary directory and then when the session exits.
The ``save_csvfiles`` function moves the CmdStan csv output files
to the specified location, renaming them using a specified basename.

.. code-block:: python

    bern_fit.save_csvfiles(dir='some/path', basename='descriptive-name')


Progress bar
------------

User can enable progress bar for the sampling if ``tqdm`` package
has been installed.

.. code-block:: python

    bern_fit = bernoulli_model.sample(data=bernoulli_data, show_progress=True)

On Jupyter Notebook environment user should use notebook version
by using ``show_progress='notebook'``.

.. code-block:: python

    bern_fit = bernoulli_model.sample(data=bernoulli_data, show_progress='notebook')

To enable javascript progress bar on Jupyter Lab Notebook user needs to install
nodejs and ipywidgets. Following the instructions in
`tqdm issue #394 <https://github.com/tqdm/tqdm/issues/394#issuecomment-384743637>`
For ``conda`` users installing nodejs can be done with ``conda``.

.. code-block:: bash

    conda install nodejs

After nodejs has been installed, user needs to install ipywidgets and enable it.

.. code-block:: bash

    pip install ipywidgets
    jupyter nbextension enable --py widgetsnbextension

Jupyter Lab still needs widgets manager.

.. code-block:: bash

    jupyter labextension install @jupyter-widgets/jupyterlab-manager
