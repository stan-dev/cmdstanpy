"Hello, World"
______________

Bayesian estimation via Stan's HMC-NUTS sampler 
------------------------------------------------

To exercise the essential functions of CmdStanPy, we will
compile the example Stan model ``bernoulli.stan``, which is
distributed with CmdStan and then fit the model to example data
``bernoulli.data.json``, also distributed with CmdStan using
Stan's HMC-NUTS sampler in order to estimate the posterior probability
of the model parameters conditioned on the data.


Specify a Stan model
^^^^^^^^^^^^^^^^^^^^

The ``CmdStanModel`` class specifies the Stan program and its corresponding compiled executable.
By default, the Stan program is compiled on instantiation.

.. code-block:: python

    import os
    from cmdstanpy import cmdstan_path, CmdStanModel

    bernoulli_stan = os.path.join(cmdstan_path(), 'examples', 'bernoulli', 'bernoulli.stan')
    bernoulli_model = CmdStanModel(stan_file=bernoulli_stan)

The ``CmdStanModel`` class provides properties and functions to inspect the model code and filepaths.

.. code-block:: python

    bernoulli_model.name
    bernoulli_model.stan_file
    bernoulli_model.exe_file
    bernoulli_model.code()


            
Run the HMC-NUTS sampler
^^^^^^^^^^^^^^^^^^^^^^^^

The ``CmdStanModel`` method ``sample`` runs the Stan HMC-NUTS sampler on the model and data
and returns a ``CmdStanMCMC`` object:

.. code-block:: python

    bernoulli_data = { "N" : 10, "y" : [0,1,0,0,0,0,0,0,0,1] }
    bern_fit = bernoulli_model.sample(data=bernoulli_data, output_dir='.')

By default, the ``sample`` command runs 4 sampler chains.
The ``output_dir`` argument specifies the path to the sampler output files.

The object returned by the ``sample`` command is a ``CmdStanMLE`` object.
It records the CmdStan arguments and command, the console outputs,
the return code, and the paths to all output files.

Because this example specifies the output dir, the output
`Stan CSV files <https://mc-stan.org/docs/cmdstan-guide/stan-csv.html#mcmc-sampler-csv-output>`__
are saved in the current working directory, one CSV file per chain.
The filenames follow the template '<model_name>-<YYYYMMDDHHMM>-<chain_id>'
plus the file suffix '.csv'.
There is also a correspondingly named file with suffix '.txt'
which contains all messages written to the console.

By default, output files are written to a temporary directory which is deleted
when the current Python session is terminated.
This is useful during model testing and development.


Access the sample
^^^^^^^^^^^^^^^^^

The ``sample`` command returns a ``CmdStanMCMC`` object
which provides methods to retrieve the sampler outputs,
the arguments used to run Cmdstan, and names of the
the per-chain stan-csv output files, and per-chain console messages files.

.. code-block:: python

   print(bern_fit)

The resulting sample from the posterior is lazily instantiated
the first time that any of the properties
``sample``, ``metric``, or ``stepsize`` are accessed.
At this point the stan-csv output files are read into memory.
For large files this may take several seconds; for the example
dataset, this should take less than a second.
The ``sample`` property of the ``CmdStanMCMC`` object
is a 3-D ``numpy.ndarray`` (i.e., a multi-dimensional array)
which contains the set of all draws from all chains 
arranged as dimensions: (draws, chains, columns).

.. code-block:: python

    bern_fit.sample.shape


The ``stan_variable(var_name)`` method returns
a numpy.ndarray which contains the set of draws in the sample for the named Stan program variable.

.. code-block:: python

    bern_fit.get_variable('theta')



    
Python's index slicing operations can be used to access the information by chain.
For example, to select all draws and all output columns from the first chain,
we specify the chain index (2nd index dimension).  As arrays indexing starts at 0,
the index '0' corresponds to the first chain in the ``CmdStanMCMC``:

.. code-block:: python

    chain_1 = bern_fit.sample[:,0,:]
    chain_1.shape       # (1000, 8)
    chain_1[0]          # sample first draw:
                        # array([-7.99462  ,  0.578072 ,  0.955103 ,  2.       ,  7.       ,
                        # 0.       ,  9.44788  ,  0.0934208])

Summarize or save the results
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

CmdStan is distributed with a posterior analysis utility
` ``stansummary`` <https://mc-stan.org/docs/cmdstan-guide/stansummary.html>
that reads the outputs of all chains and computes summary statistics
on the model fit for all sampler and model parameters and quantities of interest.
The ``CmdStanMCMC`` method ``summary`` runs this utility and returns
the total joint log-probability density `lp__` plus
the model parameters and quantities of interest in a pandas.DataFrame:

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

The ``CmdStanMCMC`` method ``diagnose`` runs the CmdStan ``diagnose`` utility
and prints the output to the console.

.. code-block:: python

    bern_fit.diagnose()

The sampler output files are written to a temporary directory which
is deleted upon session exit unless the ``output_dir`` argument is specified.
The ``save_csvfiles`` function moves the CmdStan csv output files
to a specified directory without having to re-run the sampler.

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
