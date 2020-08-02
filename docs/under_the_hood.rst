Under the Hood
______________

Under the hood, CmdStanPy uses the CmdStan command line interface
to compile and fit a model to data.
The function ``cmdstan_path`` returns the path to the local CmdStan installation.
See the installation section for more details on installing CmdStan.
  

Model Compilation
-----------------

:ref:`class_cmdstanmodel` objects manage the Stan program and its corresponding
executable.
By default, a program is compiled on object instantiation.

Model compilation is carried out via the GNU Make build tool.
The CmdStan ``makefile`` contains a set of general rules which
specify the dependencies between the Stan program and the
Stan platform components and low-level libraries.
Optional behaviors can be specified by use of variables
which are passed in to the ``make`` command as name, value pairs.

Model compilation is done in two steps:

* The ``stanc`` compiler translates the Stan program to C++.
* The C++ compiler compiles the generated code and links in
  the necessary supporting libraries.

Therefore, both the ``CmdStanModel`` constructor and ``compile`` method
allow optional arguments ``stanc_options`` and ``cpp_options`` which
specify options for each compilation step in the form of a Python dictionary
which maps compiler options to appropriate values.

In particular, in order to use Stan's 
`parallelization <https://mc-stan.org/docs/2_24/cmdstan-guide/parallelization.html>`__
features, Stan programs must be compiled with the appropriate C++ compiler flags.
If you are running GPU hardware and wish to use the OpenCL framework to speed up matrix operations,
you must set the C++ compiler flag `STAN_OPENCL`.
For high-level within-chain parallelization using the Stan language `reduce_sum` function,
it's necessary to set the C++ compiler flag `STAN_THREADS`.  While any value can be used,
we recommend the value ``True``.

For example, given Stan program named 'proc_parallel.stan', you can take
advantage of both kinds of parallelization by specifying the compiler options when instantiating
the model:

.. code-block:: python

    proc_parallel_model = CmdStanModel(
        stan_file='proc_parallel.stan',
        cpp_options={"STAN_THREADS": True, "STAN_OPENCL": True},
    )



File Handling
-------------

CmdStan is file-based interface, therefore CmdStanPy
maintains the necessary files for all models, data, and
inference method results.
CmdStanPy uses the Python library ``tempfile`` module to create
a temporary directory where all input and output files are written and
which is deleted when the Python session is terminated.


Input Data
^^^^^^^^^^

When the input data for the ``CmdStanModel`` inference methods
is supplied as a Python dictionary, this data is written to disk as
the corresponding JSON object.

Output Files
^^^^^^^^^^^^

Output filenames are composed of the model name, a timestamp
in the form 'YYYYMMDDhhmm' and the chain id, plus the corresponding
filetype suffix, either '.csv' for the CmdStan output or '.txt' for
the console messages, e.g. `bernoulli-201912081451-1.csv`. Output files
written to the temporary directory contain an additional 8-character
random string, e.g. `bernoulli-201912081451-1-5nm6as7u.csv`.


When the ``output_dir`` argument to the ``CmdStanModel`` inference methods
is given, output files are written to the specified directory, otherwise
they are written to the session-specific output directory.
All fitted model objects, i.e. ``CmdStanMCMC``, ``CmdStanVB``, ``CmdStanMLE``,
and ``CmdStanGQ``, have method ``save_csvfiles`` which moves the output files
to a specified directory.
