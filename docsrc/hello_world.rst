"Hello, World"
______________

Bayesian estimation via Stan's HMC-NUTS sampler 
------------------------------------------------

To exercise the essential functions of CmdStanPy we show how to run
Stan's HMC-NUTS sampler to estimate the posterior probability
of the model parameters conditioned on the data, 
using the example Stan model ``bernoulli.stan``
and corresponding dataset ``bernoulli.data.json`` which are
distributed with CmdStan.

This is a simple model for binary data:  given a set of N observations of i.i.d. binary data
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

The data file specifies the number of observations and their values.

.. code::

   {
    "N" : 10,
    "y" : [0,1,0,0,0,0,0,0,0,1]
   }


Instantiate the Stan model, assemble the data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :ref:`class_cmdstanmodel` class manages the Stan program and its corresponding compiled executable.
It provides properties and functions to inspect the model code and filepaths.
By default, the Stan program is compiled on instantiation.

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

            
Run the HMC-NUTS sampler
^^^^^^^^^^^^^^^^^^^^^^^^

The :ref:`class_cmdstanmodel` method ``sample`` is used to do Bayesian inference
over the model conditioned on data using  using Hamiltonian Monte Carlo
(HMC) sampling. It runs Stan's HMC-NUTS sampler on the model and data and
returns a :ref:`class_cmdstanmcmc` object.  The data can be specified
either as a filepath or a Python dict; in this example, we use the
example datafile `bernoulli.data.json`:


By default, the ``sample`` command runs 4 sampler chains.
This is a set of per-chain 
`Stan CSV files <https://mc-stan.org/docs/cmdstan-guide/stan-csv.html#mcmc-sampler-csv-output>`__
The filenames follow the template '<model_name>-<YYYYMMDDHHMM>-<chain_id>'
plus the file suffix '.csv'.
There is also a correspondingly named file with suffix '.txt'
which contains all messages written to the console.
If the ``output_dir`` argument is omitted, the output files are written
to a temporary directory which is deleted when the current Python session is terminated.

.. code-block:: python

    # specify data file
    bernoulli_data = os.path.join(cmdstan_path(), 'examples', 'bernoulli', 'bernoulli.data.json')

    # fit the model 
    bern_fit = bernoulli_model.sample(data=bernoulli_data, output_dir='.') 

    # printing the object reports sampler commands, output files
    print(bern_fit)


Access the sample
^^^^^^^^^^^^^^^^^

The :ref:`class_cmdstanmcmc` object
provides properties and methods to access, summarize, and manage the sample and its metadata.

The sampler and model outputs from each chain are written out to Stan CSV files.
The CmdStanMCMC object assembles these outputs into a
numpy.ndarray which contains all across all chains arranged as (draws, chains, columns). 
The ``draws`` method returns the draws array.
By default, it returns the underlying 3D array.
The optional boolean argument ``concat_chains``, when ``True``,
will flatten the chains resulting in a 2D array.

.. code-block:: python

    bern_fit.draws().shape 
    bern_fit.draws(concat_chains=True).shape 


To work with the draws from all chains for a parameter or quantity of interest
in the model, use the ``stan_variable`` method to obtains
a numpy.ndarray which contains the set of draws in the sample for the named Stan program variable
by flattening the draws by chains into a single column:

.. code-block:: python

    draws_theta = bern_fit.stan_variable(name='theta') 
    draws_theta.shape 


The draws array contains both the sampler variables and the model
variables. Sampler variables report the sampler state and end in `__`.
To see the names and output columns for all sampler and model
variables, we call accessor functions ``sampler_vars_cols`` and
``stan_vars_cols``:

.. code-block:: python

    sampler_variables = bern_fit.sampler_vars_cols
    stan_variables = bern_fit.stan_vars_cols
    print('Sampler variables:\n{}'.format(sampler_variables)) 
    print('Stan variables:\n{}'.format(stan_variables)) 

The NUTS-HMC sampler reports 7 variables.
The Bernoulli example model contains a single variable `theta`.
                        
Summarize the results
^^^^^^^^^^^^^^^^^^^^^

CmdStan is distributed with a posterior analysis utility
`stansummary <https://mc-stan.org/docs/cmdstan-guide/stansummary.html>`__
that reads the outputs of all chains and computes summary statistics
for all sampler and model parameters and quantities of interest.
The :ref:`class_cmdstanmcmc` method ``summary`` runs this utility and returns
summaries of the total joint log-probability density **lp__** plus
all model parameters and quantities of interest in a pandas.DataFrame:

.. code-block:: python

    bern_fit.summary()

CmdStan is distributed with a second posterior analysis utility
`diagnose <https://mc-stan.org/docs/cmdstan-guide/diagnose.html>`__
which analyzes the per-draw sampler parameters across all chains
looking for potential problems which indicate that the sample
isn't a representative sample from the posterior.
The ``diagnose`` method runs this utility and prints the output to the console.

.. code-block:: python

    bern_fit.diagnose()

Save the Stan CSV files
^^^^^^^^^^^^^^^^^^^^^^^
    
The ``save_csvfiles`` function moves the CmdStan CSV output files
to a specified directory.

.. code-block:: python

    bern_fit.save_csvfiles(dir='some/path')

.. comment
  Progress bar
  ^^^^^^^^^^^^
  
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
