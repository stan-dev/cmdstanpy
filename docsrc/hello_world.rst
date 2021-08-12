.. py:currentmodule:: cmdstanpy

"Hello, World"
--------------

Fitting a Stan model using the NUTS-HMC sampler
***********************************************

In order to verify the installation and also to demonstrate
the CmdStanPy workflow, we use CmdStanPy to fit the
the example Stan model ``bernoulli.stan``
to the dataset ``bernoulli.data.json``.
The example model and data are included with the CmdStan distribution
in subdirectory `examples/bernoulli`.

The Stan model
^^^^^^^^^^^^^^

The model ``bernoulli.stan``  is a simple model for binary data:
given a set of N observations of i.i.d. binary data
`y[1] ... y[N]`, it calculates the Bernoulli chance-of-success `theta`.

.. code::

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
CmdStanPy, manages the environment variable ``CMDSTAN`` which specifies the path to
the local CmdStan installation.
The function :func:`cmdstanpy.cmdstan_path` returns the value of this environment variable.

.. code-block:: python

    # import packages
    import os
    from cmdstanpy import cmdstan_path, CmdStanModel

    # specify Stan program file 
    bernoulli_stan = os.path.join(cmdstan_path(), 'examples', 'bernoulli', 'bernoulli.stan')

    # instantiate the model; compiles the Stan program as needed.
    bernoulli_model = CmdStanModel(stan_file=bernoulli_stan)

    # inspect model object 
    print(bernoulli_model)

            
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

The :meth:`CmdStanModel.sample` method is used to do Bayesian inference
over the model conditioned on data using  using Hamiltonian Monte Carlo
(HMC) sampling. It runs Stan's HMC-NUTS sampler on the model and data and
returns a :class:`CmdStanMCMC` object.  The data can be specified
either as a filepath or a Python dictionary; in this example, we use the
example datafile `bernoulli.data.json`:

By default, the ``sample`` method runs 4 sampler chains.
The ``output_dir`` argument is an optional argument which specifies
the path to the output directory used by CmdStan.
If this argument is omitted, the output files are written
to a temporary directory which is deleted when the current Python session is terminated. [#]_


.. code-block:: python

    # specify data file
    bernoulli_data = os.path.join(cmdstan_path(), 'examples', 'bernoulli', 'bernoulli.data.json')

    # fit the model 
    bernoulli_fit = bernoulli_model.sample(data=bernoulli_data, output_dir='.') 

    # printing the object reports sampler commands, output files
    print(bernoulli_fit)


Accessing the sample
^^^^^^^^^^^^^^^^^^^^

The CmdStan `sample` method outputs are a set of per-chain
`Stan CSV files <https://mc-stan.org/docs/cmdstan-guide/stan-csv.html#mcmc-sampler-csv-output>`__.
The filenames follow the template '<model_name>-<YYYYMMDDHHMM>-<chain_id>'
plus the file suffix '.csv'.
The :class:`CmdStanMCMC` class provides methods to assemble the contents
of these files in memory as well as methods to manage the disk files.

Underlyingly, the draws from all chains are stored as an
a numpy.ndarray with dimensions: draws, chains, columns.
CmdStanPy provides accessor methods which return the sample
either in terms of the CSV file columns or in terms of the
sampler and Stan program variables.
The :meth:`~CmdStanMCMC.draws` and :meth:`~CmdStanMCMC.draws_pd` methods return the sample contents
in columnar format.

The :meth:`~CmdStanMCMC.stan_variable` method to returns a numpy.ndarray object
which contains the set of all draws in the sample for the named Stan program variable.
The draws from all chains are flattened into a single drawset.
The first ndarray dimension is the number of draws X number of chains.
The remaining ndarray dimensions correspond to the Stan program variable dimension.
The :meth:`~CmdStanMCMC.stan_variables` method returns a Python dict over all Stan model variables.

.. code-block:: python

    bernoulli_fit.draws().shape 
    bernoulli_fit.draws(concat_chains=True).shape 

    draws_theta = bernoulli_fit.stan_variable(name='theta') 
    draws_theta.shape 

                        
CmdStan utilities:  `stansummary`, `diagnose`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

CmdStan is distributed with a posterior analysis utility
`stansummary <https://mc-stan.org/docs/cmdstan-guide/stansummary.html>`__
that reads the outputs of all chains and computes summary statistics
for all sampler and model parameters and quantities of interest.
The :class:`CmdStanMCMC` method :meth:`~CmdStanMCMC.summary` runs this utility and returns
summaries of the total joint log-probability density **lp__** plus
all model parameters and quantities of interest in a pandas.DataFrame:

.. code-block:: python

    bernoulli_fit.summary()

CmdStan is distributed with a second posterior analysis utility
`diagnose <https://mc-stan.org/docs/cmdstan-guide/diagnose.html>`__
which analyzes the per-draw sampler parameters across all chains
looking for potential problems which indicate that the sample
isn't a representative sample from the posterior.
The :meth:`~CmdStanMCMC.diagnose` method runs this utility and prints the output to the console.

.. code-block:: python

    bernoulli_fit.diagnose()

Managing Stan CSV files
^^^^^^^^^^^^^^^^^^^^^^^

The :class:`CmdStanMCMC` object keeps track of all output files produced
by the sampler run.
The :meth:`~CmdStanMCMC.save_csvfiles` function moves the CSV files
to a specified directory.

.. code-block:: python

    bernoulli_fit.save_csvfiles(dir='some/path')

.. rubric:: Footnotes

.. [#] Because CmdStan is file-based and because model development
       more often than not requires many successive runs of a model,
       sending outputs to a temporary directory avoids filling up
       the filesystem with unneeded CSV files. Since more user time
       is spent in model development, not saving outputs
       is the default option.

.. comment
  Progress bar
  ^^^^^^^^^^^^
  
  User can enable progress bar for the sampling if ``tqdm`` package
  has been installed.
  
  .. code-block:: python
  
      bernoulli_fit = bernoulli_model.sample(data=bernoulli_data, show_progress=True)
  
  On Jupyter Notebook environment user should use notebook version
  by using ``show_progress='notebook'``.
  
  .. code-block:: python
  
      bernoulli_fit = bernoulli_model.sample(data=bernoulli_data, show_progress='notebook')
  
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

      
