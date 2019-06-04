Guide
=====

This tutorial will walk you through the different steps of using CmdStanPy. 

Install
_______

CmdStanPy can be installed from GitHub

.. code-block:: bash

	pip install -e git+https://github.com/stan-dev/cmdstanpy

CmdStanPy requires a local install of CmdStan.
If you don't have CmdStan installed, you can run the script ``make_cmdstan.sh`` which
will download CmdStan from GitHub and build the CmdStan utilities.
By default this script installs the latest version of CmdStan into a directory named
`.cmdstanpy` in the user's `$HOME` directory:


.. code-block:: bash

	./make_cmdstan.sh
    ls -F ~/.cmdstanpy

This script takes two optional named arguments: `-d <directory> -v <version>`

.. code-block:: bash

	./make_cmdstan.sh -d cmdstan -v 2.18.1
    ls -F ~/cmdstan

If you already have CmdStan installed, then set the ``CMDSTAN`` environment variable accordingly,
either from within your Python session or setting the environment variable directly using ``bash``:

.. code-block:: bash

	export CMDSTAN='/path/to/cmdstan'

.. code-block:: python

	import os
	os.environ['CMDSTAN'] = '/path/to/cmdstan'


Basics
______

The ``compile_model`` function takes as its argument the name of the Stan program and returns a ``Model`` object:

.. code-block:: python

	import os
	import os.path
	from cmdstanpy.lib import Model
	from cmdstanpy.cmds import compile_model
	bernoulli_stan = os.path.join('cmdstanpy', 'test', 'data', 'bernoulli.stan')
	bernoulli_model = compile_model(bernoulli_stan)
	print(bernoulli_model)
	bernoulli_model.name

The ``Model`` class specifies the Stan program and its corresponding compiled executable.
If you already have a compiled executable, you can construct the Model object directly:

.. code-block:: python

	bernoulli_model = Model(
			stan_file=os.path.join('cmdstanpy', 'test', 'data', 'bernoulli.stan'),
			stan_exe=os.path.join('cmdstanpy', 'test', 'data', 'bernoulli')
		    )
	print(bernoulli_model)
	bernoulli_model.name

The ``sample`` function invokes the Stan HMC-NUTS sampler on the ``Model`` object and some data
and returns a ``RunSet`` object:

.. code-block:: python

    bern_data = { "N" : 10, "y" : [0,1,0,0,0,0,0,0,0,1] }
    bern_fit = sample(bernoulli_model, chains=4, cores=2, data=bern_data)

The ``sample`` property of the ``RunSet`` object is a 3-D ``numpy.ndarray``
which contains all draws across all chains, stored column major format so that values
for each parameter are stored contiguously in memory.
The dimensions of the ndarray are arranged (draws, chains, columns).

The ``get_drawset`` function flattens this 3-D ndarray to a pandas.DataFrame,
one draw per row.  The `params` argument is used to restrict the DataFrame
view to the specified parameter names, else all output columns are returned.

.. code-block:: python

    bern_fit.sample.shape
    get_drawset(bern_fit, params=['theta'])


CmdStan is distributed with a posterior analysis utility `stansummary`
that reads the outputs of all chains and computes summary statistics
on the model fit for all parameters. CmdStanPy's ``summary`` function
runs this utility and returns the output as a pandas.DataFrame:

.. code-block:: python

    summary(bern_fit)

CmdStan is distributed with a second posterior analysis utility `diagnose`
that reads the outputs of all chains and checks for the following
potential problems:

+ Transitions that hit the maximum treedepth
+ Divergent transitions
+ Low E-BFMI values (sampler transitions HMC potential energy)
+ Low effective sample sizes
+ High R-hat values

The ``diagnose`` function prints the output of the CmdStan ``bin/diagnose``:

.. code-block:: python

    diagnose(bern_fit)

By default, CmdStanPy will save all CmdStan outputs in a temporary
directory which is deleted when the Python session exits.
In particular, if the ``sample`` command is invoked without
specifying the `csv_output_file` path, then the csv output files
will be written into this temporary directory and therefore will
be deleted once the session exits.
The ``save_csvfiles`` function moves the CmdStan csv output files
to the specified location, renaming them using a specified basename.

.. code-block:: python

    save_csvfiles(bern_fit, dir='some/path', basename='descriptive-name')

