Guide
=====

This tutorial will walk you through the different steps of using CmdStanPy. 

Install
-------

CmdStanPy can be installed from GitHub

.. code-block:: bash

	pip install -e git+https://github.com/stan-dev/cmdstanpy

CmdStanPy requires a local install of CmdStan.
If you don't have CmdStan installed, the script ``make_cmdstan.sh`` will install the latest version
of CmdStan in the ``releases`` directory.  

.. code-block:: bash

	./make_cmdstan.sh

If you already have CmdStan installed, then set the ``CMDSTAN`` environment variable accordingly,
either from within your Python session or setting the environment variable directly using ``bash``:

.. code-block:: bash

	export CMDSTAN='/path/to/cmdstan'

.. code-block:: python

	import os
	os.environ['CMDSTAN'] = '/path/to/cmdstan'


Basics
------

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
and returns a ``PosteriorSample`` object:

.. code-block:: python

    bern_data = { "N" : 10, "y" : [0,1,0,0,0,0,0,0,0,1] }
    bern_sample = sample(bernoulli_model, chains=4, cores=2, data=bern_data)

The ``sample`` property of the ``PosteriorSample`` object is a 3-D ``numpy.ndarray``
which contains all draws across all chains, stored column major format so that values
for each parameter are stored contiguously in memory.
The dimensions of the ndarray are arranged (draws, chains, columns).
The ``extract`` function flattens this 3-D ndarray to a pandas.DataFrame,
one draw per row.  The `params` argument is used to restrict the DataFrame
view to the specified parameter names, else all output columns are returned.

.. code-block:: python

    bern_sample.sample.shape
    bern_sample.extract(params=['theta'])


A ``PosteriorSample`` object's ``summary`` function returns the output of the CmdStan ``bin/stansummary``
utility as pandas.DataFrame:

.. code-block:: python

    bern_sample.summary()


