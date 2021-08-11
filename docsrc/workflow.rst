CmdStanPy Workflow
__________________


Developing a good analysis of a problem or dataset is an
open-ended process, usually requiring many iterations of
developing a model, fitting the data, and evaluating the results.
The Bayesian workflow for model comparison and model expansion
provides a framework for organizing these activities.

CmdStanPy provides the functionality needed to
compile the model and assemble the data, do inference,
and to validate, access, and export the resulting inference data.
For each CmdStan inference method there is a CmdStanPy class which manages
the resulting inference method outputs:

* method `sample` returns a `CmdStanMCMC` object

* method `optimize` returns a `CmdStanMLE` object

* method `variational` returns a `CmdStanVB` object

* method `generate` returns a `CmdStanGQ` object

These objects have a common set of methods for accessing the
inference results and metadata, as well as method-specific properties.

The following sections describe how to carry out one iteration
of the Bayesian workflow using CmdStanPy.
During the course of an analysis, it is expected that this sequence
of commands will be carried out repeatedly,
either to compare different models or to compare the results of
difference infenence methods.


Specifying a Stan model
^^^^^^^^^^^^^^^^^^^^^^^

The: :ref:`class_cmdstanmodel` class manages the Stan program and its corresponding compiled executable and
provides properties and functions to inspect the model code and filepaths.

A model object can be instantiated by specifying either the Stan program file path
or the compiled executable file path or both.
If only the Stan program file is specified, by default,
CmdStanPy will try to compile the model.
If both the model and executable file are specified,
the constructor will compare the filesystem timestamps and
will only compile the program if the Stan file has a later timestamp which
indicates that the program may have been edited.
The constructor argument `compile=False` will override the default behavoir.

.. code-block:: python

    import os
    from cmdstanpy import CmdStanModel

    my_stanfile = os.path.join('.', 'my_model.stan')
    my_model = CmdStanModel(stan_file=my_stanfile)
    my_model.name
    my_model.stan_file
    my_model.exe_file
    my_model.code()

.. _model-compilation:

The `CmdStanModel` class provides a `compile` method which can be used to
recompile the model as needed and and the argument `force=True` can be used
to insure that the Stan program is recompiled, even if the timestamp on the
executable file is new than the timestamp on the Stan program file.
 
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
`parallelization <https://mc-stan.org/docs/cmdstan-guide/parallelization.html>`__
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



Assembling model data inputs and initializations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

CmdStan is file-based interface, therefore all model input and
initialization data must be supplied as JSON files, as described in the
CmdStan User's Guide:
<https://mc-stan.org/docs/cmdstan-guide/json.html>.

CmdStanPy inference methods allow inputs and initializations
to be specified as in-memory Python dictionary objects
which are then converted to JSON via the utility method `write_stan_json`.
This method is available as part of the CmdStanPy API.
It should be used to create JSON input files whenever
these inputs contain either a collection compatible with
numpy arrays or pandas.Series.


Doing inference
^^^^^^^^^^^^^^^


List of inference methods documentation, 1 sentence summary of each

  
Working with the inference engine outputs
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Common accessor methods


Per method outputs















"hello world" outtakes
""""""""""""""""""""""
CmdStanPy also saves all CmdStan messages and error messages into 
files with the same template basename and with suffix '.txt' and '.err', respectively.


Information from the Stan CSV files header comments and header row
is parsed into a :ref:`class_inferencemetadata` object which
can be accessed via the `CmdStanMCMC` object's `metadata` property.
The ``stan_variables`` method returns a Python dict over all Stan model variables.
The ``method_variables`` method returns a Python dict over all NUTS-HMC sampler method
output variables.


Information from the Stan CSV files header comments and header row
is parsed into a :ref:`class_inferencemetadata` object which
can be accessed via the `CmdStanMCMC` object's `metadata` property.
The NUTS-HMC sampler reports both the estimates for all variables
in the Stan program's `parameter`, `transformed parameter`, and `generated quantities` 
NUTS-HMC sampler 
The draws array contains both the sampler method variables
and the model variables. The sampler method variables report
the sampler state.  All method variables end in `__`.
The `InferenceMetadata` properties ``method_vars_cols``
and ``stan_vars_cols`` map the method and model variable
names to the column or columns that they span.

.. code-block:: python

    sampler_variables = bernoulli_fit.metadata.method_vars_cols
    stan_variables = bernoulli_fit.metadata.stan_vars_cols
    print('Sampler variables:\n{}'.format(sampler_variables)) 
    print('Stan variables:\n{}'.format(stan_variables)) 

The NUTS-HMC sampler reports 7 variables.
The Bernoulli example model contains a single variable `theta`.
