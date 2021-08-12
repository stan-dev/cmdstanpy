CmdStanPy Workflow
__________________


Developing a good analysis of a problem or dataset is an
open-ended process, usually requiring many iterations of
developing a model, fitting the data, and evaluating the results.
The Bayesian workflow for model comparison and model expansion
provides a framework for organizing these activities.

CmdStanPy provides the tools needed to
compile a Stan model, assemble input data,
do inference on the model conditioned on the data,
and validate, access, and export the results.
The following sections describe the process of building, running, and
managing the resulting inference for a single model and set of inputs.

During the course of an analysis, it is expected that this sequence
of commands will be carried out repeatedly,
either to compare different models or to compare the results of
difference infenence methods.

.. _model-compilation:

Compile the Stan model
^^^^^^^^^^^^^^^^^^^^^^

The: :class:`cmdstanpy.CmdStanModel` class manages the Stan program and its corresponding compiled executable and
provides properties and functions to inspect the model code and filepaths.

A model object can be instantiated by specifying either the Stan program file path
or the compiled executable file path or both.
If only the Stan program file is specified, by default,
CmdStanPy will try to compile the model.
If both the model and executable file are specified,
the constructor will compare the filesystem timestamps and
will only compile the program if the Stan file has a later timestamp which
indicates that the program may have been edited.
The constructor argument ``compile=False`` will override the default behavoir.

.. code-block:: python

    import os
    from cmdstanpy import CmdStanModel

    my_stanfile = os.path.join('.', 'my_model.stan')
    my_model = CmdStanModel(stan_file=my_stanfile)
    my_model.name
    my_model.stan_file
    my_model.exe_file
    my_model.code()

The method :meth:`cmdstanpy.CmdStanModel.compile` is used to compile the model as needed.
When the argument ``force=True`` is present, CmdStanPy will always compile the model,
even if the existing executable file is newer than the Stan program file.
 
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


Assemble input and initialization data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

CmdStan is file-based interface, therefore all model input and
initialization data must be supplied as JSON files, as described in the
CmdStan User's Guide:
<https://mc-stan.org/docs/cmdstan-guide/json.html>.

CmdStanPy inference methods allow inputs and initializations
to be specified as in-memory Python dictionary objects
which are then converted to JSON via the utility method :func:`cmdstanpy.write_stan_json`.
This method should be used to create JSON input files whenever
these inputs contain either a collection compatible with
numpy arrays or pandas.Series.


Run the CmdStan inference engine
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For each CmdStan inference method, there is a corresponding method on the CmdStanModel class.

* The :meth:`cmdstanpy.CmdStanModel.sample` method runs Stan's
  `HMC-NUTS sampler <https://mc-stan.org/docs/reference-manual/hamiltonian-monte-carlo.html>`_
  and returns a :class:`cmdstanpy.CmdStanMCMC` object which contains
  a sample from the posterior distribution of the model conditioned on the data.

* The :meth:`cmdstanpy.CmdStanModel.variational` method runs Stan's
  `Automatic Differentiation Variational Inference (ADVI) algorithm <https://mc-stan.org/docs/reference-manual/vi-algorithms-chapter.html>`_
  and returns   a :class:`cmdstanpy.CmdStanVB` object which contains
  an approximation the posterior distribution in the unconstrained variable space.

* The :meth:`cmdstanpy.CmdStanModel.optimize` runs one of
  `Stan's optimization algorithms <https://mc-stan.org/docs/reference-manual/optimization-algorithms-chapter.html>`_
  to find a mode of the density specified by the Stan program.  It
  returns a :class:`cmdstanpy.CmdStanMLE` object.

* The :meth:`cmdstanpy.CmdStanModel.generate` method runs Stan's
  `generate_quantities method <https://mc-stan.org/docs/cmdstan-guide/standalone-generate-quantities.html>`_
  which generates additional quantities of interest from a mode. Its take an existing sample as input and
  uses the parameter estimates in the sample to run the Stan program's ``generated quantities`` block.
  It returns a :class:`cmdstanpy.CmdStanGQ` object.

  
Validate, view, export the inference engine outputs
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The inference engine results objects 
``CmdStanMCMC``, ``CmdStanVB``, ``CmdStanMLE`` and ``CmdStanGQ``
contain the CmdStan method configuration information
and the location of all output files produced.
The provide a common set methods for accessing the inference results and metadata,
as well as method-specific informational properties and methods.objects 


By `metadata` we mean the information parsed from the header comments and header row of the
`Stan CSV files <https://mc-stan.org/docs/cmdstan-guide/stan-csv.html>`_.
into a :class:`cmdstanpy.InferenceMetadata` object which is exposed via
the object's `metadata` property.

* The metadata :meth:`cmdstanpy.InferenceMetadata.cmdstan_config`
  property provides the CmdStan configuration information parsed out
  of the Stan CSV file header.

* The metadata :meth:`cmdstanpy.InferenceMetadata.method_vars_cols`
  property returns the names, column indices of the inference engine method variables,
  e.g.,
  `the NUTS-HMC sampler output variables <https://mc-stan.org/docs/cmdstan-guide/mcmc-intro.html#mcmc_output_csv>`_
  are ``lp__``, ..., ``energy__``.

* The metadata :meth:`cmdstanpy.InferenceMetadata.stan_vars_cols`
  property returns the names, column indices of all Stan model variables.
  Container variables will span as many columns, one column per element.

* The metadata :meth:`cmdstanpy.InferenceMetadata.stan_vars_dims`
  property specifies the names, dimensions of the Stan model variables.

The CSV data is assembled into the inference result object.
CmdStanPy provides accessor methods which return this information
either as columnar data (i.e., in terms of the CSV file columns),
or as method and model variables.

The ``draws`` and ``draws_pd`` methods return the sample contents
in columnar format, as a numpy.ndarray or pandas.DataFrame, respectively.

The ``stan_variable`` method to returns a numpy.ndarray object
which contains the set of all draws in the sample for the named Stan program variable.
The draws from all chains are flattened into a single drawset.
The first ndarray dimension is the number of draws X number of chains.
The remaining ndarray dimensions correspond to the Stan program variable dimension.
The ``stan_variables`` method returns a Python dict over all Stan model variables.

The ``model_variables`` method returns a Python dict over all inference
method variables.

The ``draws_xr`` method returns the sample contents as an
`xarray.DataSet <http://xarray.pydata.org/en/stable/generated/xarray.Dataset.html>`_
which maps the method and model variable names to their respective values.
