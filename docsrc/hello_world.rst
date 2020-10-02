"Hello, World"
______________

Bayesian estimation via Stan's HMC-NUTS sampler 
------------------------------------------------

To exercise the essential functions of CmdStanPy we show how to run
Stan's HMC-NUTS sampler to estimate the posterior probability
of the model parameters conditioned on the data.
Do do this we use the example Stan model ``bernoulli.stan``
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


Specify a Stan model
^^^^^^^^^^^^^^^^^^^^

The :ref:`class_cmdstanmodel` class manages the Stan program and its corresponding compiled executable.
It provides properties and functions to inspect the model code and filepaths.
By default, the Stan program is compiled on instantiation.

.. code-block:: python

    import os
    from cmdstanpy import cmdstan_path, CmdStanModel

    bernoulli_stan = os.path.join(cmdstan_path(), 'examples', 'bernoulli', 'bernoulli.stan')
    bernoulli_model = CmdStanModel(stan_file=bernoulli_stan)
    bernoulli_model.name
    bernoulli_model.stan_file
    bernoulli_model.exe_file
    bernoulli_model.code()


            
Run the HMC-NUTS sampler
^^^^^^^^^^^^^^^^^^^^^^^^

The :ref:`class_cmdstanmodel` method ``sample`` is used to do Bayesian inference
over the model conditioned on data using  using Hamiltonian Monte Carlo
(HMC) sampling. It runs Stan's HMC-NUTS sampler on the model and data and
returns a :ref:`class_cmdstanmcmc` object.

.. code-block:: python

    bernoulli_data = os.path.join(cmdstan_path(), 'examples', 'bernoulli', 'bernoulli.data.json')
    bern_fit = bernoulli_model.sample(data=bernoulli_data, output_dir='.')

By default, the ``sample`` command runs 4 sampler chains.
This is a set of per-chain 
`Stan CSV files <https://mc-stan.org/docs/cmdstan-guide/stan-csv.html#mcmc-sampler-csv-output>`__
The filenames follow the template '<model_name>-<YYYYMMDDHHMM>-<chain_id>'
plus the file suffix '.csv'.
There is also a correspondingly named file with suffix '.txt'
which contains all messages written to the console.
If the ``output_dir`` argument is omitted, the output files are written
to a temporary directory which is deleted when the current Python session is terminated.


Access the sample
^^^^^^^^^^^^^^^^^

The :ref:`class_cmdstanmcmc` object stores the CmdStan config information and
the names of the the per-chain output files.
It manages and retrieves the sampler outputs as Python objects.

.. code-block:: python

   print(bern_fit)

The resulting set of draws produced by the sampler is lazily instantiated
as a 3-D ``numpy.ndarray`` (i.e., a multi-dimensional array)
over all draws from all chains  arranged as draws X chains X columns.
Instantiation happens the first time that any of the information
in the posterior is accesed via properties:
``draws``, ``metric``, or ``stepsize`` are accessed.
At this point the stan-csv output files are read into memory.
For large files this may take several seconds; for the example
dataset, this should take less than a second.

.. code-block:: python

    bern_fit.draws().shape
    
Python's index slicing operations can be used to access the information by chain.
For example, to select all draws and all output columns from the first chain,
we specify the chain index (2nd index dimension).  As arrays indexing starts at 0,
the index '0' corresponds to the first chain in the :ref:`class_cmdstanmcmc`:

.. code-block:: python

    chain_1 = bern_fit.draws()[:,0,:]
    chain_1.shape       # (1000, 8)
    chain_1[0]          # first draw:
                        # array([-7.99462  ,  0.578072 ,  0.955103 ,  2.       ,  7.       ,
                        # 0.       ,  9.44788  ,  0.0934208])

To work with the draws from all chains for a parameter or quantity of interest
in the model, use the ``stan_variable`` method to obtains
a numpy.ndarray which contains the set of draws in the sample for the named Stan program variable
by flattening the draws by chains into a single column:

.. code-block:: python

    bern_fit.stan_variable('theta')

                        
Summarize or save the results
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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

The ``save_csvfiles`` function moves the CmdStan csv output files
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
