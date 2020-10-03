Stan Models in CmdStanPy
________________________

The: :ref:`class_cmdstanmodel` class manages the Stan program and its corresponding compiled executable.
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


A model object can be instantiated by specifying either the Stan program file path
or the compiled executable file path or both.
If both are specified, the constructor will check the timestamps on each and
will only re-compile the program if the Stan file has a later timestamp which
indicates that the program may have been edited.

.. _model-compilation:

Model compilation
-----------------

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

Therefore, both the constructor and the ``compile`` method
allow optional arguments ``stanc_options`` and ``cpp_options`` which
specify options for each compilation step.
Options are specified as a Python dictionary mapping
compiler option names to appropriate values.

To use Stan's 
`parallelization <https://mc-stan.org/docs/2_24/cmdstan-guide/parallelization.html>`__
features, Stan programs must be compiled with the appropriate C++ compiler flags.
If you are running GPU hardware and wish to use the OpenCL framework to speed up matrix operations,
you must set the C++ compiler flag **STAN_OPENCL**.
For high-level within-chain parallelization using the Stan language `reduce_sum` function,
it's necessary to set the C++ compiler flag **STAN_THREADS**.  While any value can be used,
we recommend the value ``True``.

For example, given Stan program named 'proc_parallel.stan', you can take
advantage of both kinds of parallelization by specifying the compiler options when instantiating
the model:

.. code-block:: python

    proc_parallel_model = CmdStanModel(
        stan_file='proc_parallel.stan',
        cpp_options={"STAN_THREADS": True, "STAN_OPENCL": True},
    )


Specifying a custom Make tool
"""""""""""""""""""""""""""""

To use custom Make-tool use ``set_make_env`` function.

.. code-block:: python

    from cmdstanpy import set_make_env
    set_make_env("mingw32-make.exe") # On Windows with mingw32-make
