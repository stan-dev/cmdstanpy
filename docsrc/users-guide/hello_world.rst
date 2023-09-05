.. py:currentmodule:: cmdstanpy

"Hello, World!"
---------------

Fitting a Stan model using the NUTS-HMC sampler
***********************************************

In order to verify the installation and also to demonstrate
the CmdStanPy workflow, we use CmdStanPy to fit the
the example Stan model ``bernoulli.stan``
to the dataset ``bernoulli.data.json``.
The ``bernoulli.stan`` is a `Hello, World! <https://en.wikipedia.org/wiki/%22Hello,_World!%22_program>`__
program which illustrates the basic syntax of the Stan language.
It allows the user to verify that CmdStanPy, CmdStan,
the StanC compiler, and the C++ toolchain have all been properly installed.

For substantive example models and
guidance on coding statistical models in Stan, see
the `Stan User's Guide <https://mc-stan.org/docs/stan-users-guide/index.html>`_.


The Stan model
^^^^^^^^^^^^^^

The model ``bernoulli.stan``  is a trivial model:
given a set of N observations of i.i.d. binary data
`y[1] ... y[N]`, it calculates the Bernoulli chance-of-success `theta`.

.. code:: stan

    data {
      int<lower=0> N;
      int<lower=0,upper=1> y[N];
    }
    parameters {
      real<lower=0,upper=1> theta;
    }
    model {
      theta ~ beta(1,1);  // uniform prior on interval 0,1
      y ~ bernoulli(theta);
    }

The :class:`CmdStanModel` class manages the Stan program and its corresponding compiled executable.
It provides properties and functions to inspect the model code and filepaths.
A `CmdStanModel` can be instantiated from a Stan file or its corresponding compiled executable file.

.. ipython:: python

    # import packages
    import os
    from cmdstanpy import CmdStanModel

    # specify Stan program file
    stan_file = os.path.join('users-guide', 'examples', 'bernoulli.stan')

    # instantiate the model object
    model = CmdStanModel(stan_file=stan_file)

    # inspect model object
    print(model)

    # inspect compiled model
    print(model.exe_info())



Data inputs
^^^^^^^^^^^

CmdStanPy accepts input data either as a Python dictionary which maps data variable names
to values, or as the corresponding JSON file.

The bernoulli model requires two inputs: the number of observations `N`, and
an N-length vector `y` of binary outcomes.
The data file `bernoulli.data.json` contains the following inputs:

.. code::

   {
    "N" : 10,
    "y" : [0,1,0,0,0,0,0,0,0,1]
   }



Fitting the model
^^^^^^^^^^^^^^^^^

The :meth:`~CmdStanModel.sample` method is used to do Bayesian inference
over the model conditioned on data using  using Hamiltonian Monte Carlo
(HMC) sampling. It runs Stan's HMC-NUTS sampler on the model and data and
returns a :class:`CmdStanMCMC` object.  The data can be specified
either as a filepath or a Python dictionary; in this example, we use the
example datafile `bernoulli.data.json`:
By default, the `sample` method runs 4 sampler chains.


.. ipython:: python

    # specify data file
    data_file = os.path.join('users-guide', 'examples', 'bernoulli.data.json')

    # fit the model
    fit = model.sample(data=data_file)


*Note* this model can be fit using other methods

+ the :meth:`~CmdStanModel.variational` method does approximate Bayesian inference and returns a :class:`CmdStanVB` object
+ the :meth:`~CmdStanModel.optimize` method does maximum likelihood estimation and returns a :class:`CmdStanMLE` object

Accessing the results
^^^^^^^^^^^^^^^^^^^^^

The sampler outputs are the set of per-chain
`Stan CSV files <https://mc-stan.org/docs/cmdstan-guide/stan-csv.html>`_,
a non-standard CSV file format.
Each data row of the Stan CSV file contains the per-iteration estimate of the Stan model
parameters, transformed parameters,  and generated quantities variables.
Container variables, i.e., vector, row-vector, matrix, and array variables
are necessarily serialized into a single row's worth of data.
The output objects parse the set of Stan CSV files into  a set of in-memory data structures
and provide accessor functions for the all estimates and metadata.
CmdStanPy makes a distinction between the per-iteration model outputs
and the per-iteration algorithm outputs:  the former are 'stan_variables'
and the latter are 'method_variables'.

The `CmdStanMCMC` object provides the following accessor methods:

+ :meth:`~CmdStanMCMC.stan_variable`: returns an numpy.ndarray whose structure corresponds to the Stan program variable structure

+ :meth:`~CmdStanMCMC.stan_variables`: returns an Python dictionary mapping the Stan program variable names to the corresponding numpy.ndarray.

+ :meth:`~CmdStanMCMC.draws`:  returns a numpy.ndarray which is either a 3-D array draws X chains X CSV columns,
  or a 2-D array draws X columns, where the chains are concatenated into a single column.
  The argument `vars` can be used to restrict this to just the columns for one or more variables.

+ :meth:`~CmdStanMCMC.draws_pd`: returns a pandas.DataFrame over all columns in the Stan CSV file.
  The argument `vars` can be used to restrict this to one or more variables.

+ :meth:`~CmdStanMCMC.draws_xr`: returns an xarray.Dataset which maps model variable names to their respective values.
  The argument `vars` can be used to restrict this to one or more variables.

+ :meth:`~CmdStanMCMC.method_variables`: returns a Python dictionary over the sampler diagnostic/information output columns
  which by convention end in ``__``, e.g., ``lp__``.


.. ipython:: python

    # access model variable by name
    print(fit.stan_variable('theta'))
    print(fit.draws_pd('theta')[:3])
    print(fit.draws_xr('theta'))
    # access all model variables
    for k, v in fit.stan_variables().items():
        print(f'{k}\t{v.shape}')
    # access the sampler method variables
    for k, v in fit.method_variables().items():
        print(f'{k}\t{v.shape}')
    # access all Stan CSV file columns
    print(f'numpy.ndarray of draws: {fit.draws().shape}')
    fit.draws_pd()


In addition to the MCMC sample itself, the CmdStanMCMC object provides
access to the the per-chain HMC tuning parameters from the NUTS-HMC adaptive sampler,
(if present).

.. ipython:: python

    print(fit.metric_type)
    print(fit.metric)
    print(fit.step_size)


The CmdStanMCMC object also provides access to metadata about the model and the sampler run.

.. ipython:: python

    print(fit.metadata.cmdstan_config['model'])
    print(fit.metadata.cmdstan_config['seed'])


CmdStan utilities:  ``stansummary``, ``diagnose``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

CmdStan is distributed with a posterior analysis utility
`stansummary <https://mc-stan.org/docs/cmdstan-guide/stansummary.html>`__
that reads the outputs of all chains and computes summary statistics
for all sampler and model parameters and quantities of interest.
The :class:`CmdStanMCMC` method :meth:`~CmdStanMCMC.summary` runs this utility and returns
summaries of the total joint log-probability density **lp__** plus
all model parameters and quantities of interest in a pandas.DataFrame:

.. ipython:: python

    fit.summary()


CmdStan is distributed with a second posterior analysis utility
`diagnose <https://mc-stan.org/docs/cmdstan-guide/diagnose.html>`__
which analyzes the per-draw sampler parameters across all chains
looking for potential problems which indicate that the sample
isn't a representative sample from the posterior.
The :meth:`~CmdStanMCMC.diagnose` method runs this utility and prints the output to the console.

.. ipython:: python

    print(fit.diagnose())
