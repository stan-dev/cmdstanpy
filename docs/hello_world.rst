"Hello World"
_____________

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

The ``Model`` class specifies the Stan program and its corresponding compiled executable.
The method ``compile`` is used to compile or or recompile a Stan program.

.. code-block:: python

    import os
    from cmdstanpy import cmdstan_path, Model

    bernoulli_stan = os.path.join(cmdstan_path(), 'examples', 'bernoulli', 'bernoulli.stan')
    bernoulli_model = Model(stan_file=bernoulli_stan)
    bernoulli_model.compile()

If you already have a compiled executable, you can construct a ``Model`` object directly:

.. code-block:: python

    bernoulli_model = Model(
            stan_file=os.path.join(cmdstan_path(), 'examples', 'bernoulli', 'bernoulli.stan'),
            exe_file=os.path.join(cmdstan_path(), 'examples', 'bernoulli', 'bernoulli')
            )

            
Run the HMC-NUTS sampler
^^^^^^^^^^^^^^^^^^^^^^^^

The ``Model`` method ``sample`` runs the Stan HMC-NUTS sampler on the model and data
and returns a ``StanMCMC`` object:

.. code-block:: python

    bernoulli_data = { "N" : 10, "y" : [0,1,0,0,0,0,0,0,0,1] }
    bern_fit = bernoulli_model.sample(data=bernoulli_data, csv_basename='./bern')

By default, the ``sample`` command runs 4 sampler chains.
The ``csv_basename`` argument specifies the path and filename prefix
of the sampler output files.
If no output file path is specified, the sampler outputs
are written to a temporary directory which is deleted
when the current Python session is terminated.


Access the sample
^^^^^^^^^^^^^^^^^

The ``sample`` command returns a ``StanMCMC`` object
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
The ``sample`` property of the ``StanMCMC`` object
is a 3-D ``numpy.ndarray`` (i.e., a multi-dimensional array)
which contains the set of all draws from all chains 
arranged as dimensions: (draws, chains, columns).

.. code-block:: python

    bern_fit.sample.shape


The ``get_drawset`` method returns the draws from
all chains as a ``pandas.DataFrame``, one draw per row, one column per
model parameter, transformed parameter, generated quantity variable.
The ``params`` argument is used to restrict the DataFrame
columns to just the specified parameter names.

.. code-block:: python

    bern_fit.get_drawset(params=['theta'])

Python's index slicing operations can be used to access the information by chain.
For example, to select all draws and all output columns from the first chain,
we specify the chain index (2nd index dimension).  As arrays indexing starts at 0,
the index '0' corresponds to the first chain in the ``StanMCMC``:

.. code-block:: python

    chain_1 = bern_fit.sample[:,0,:]
    chain_1.shape       # (1000, 8)
    chain_1[0]          # sample first draw:
                        # array([-7.99462  ,  0.578072 ,  0.955103 ,  2.       ,  7.       ,
                        # 0.       ,  9.44788  ,  0.0934208])

Summarize or save the results
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

CmdStan is distributed with a posterior analysis utility ``stansummary``
that reads the outputs of all chains and computes summary statistics
on the model fit for all parameters. The ``StanMCMC`` method ``summary``
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

The ``StanMCMC`` method ``diagnose`` runs the CmdStan ``diagnose`` utility
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
